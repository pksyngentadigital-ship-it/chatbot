"""Pinecone access: QueryPlan -> Evidence.

One retrieval path serves every mode. The old code had three that had
drifted apart; the differences between them were bugs, not features.
"""

from dataclasses import dataclass, field

from vog import llm
from vog import parsing as P
from vog.catalog import EMBEDDING_DIMENSION, INDEX_STATS_ID, PINECONE_INDEX_NAME
from vog.plan import QueryPlan, Segment

# A zero vector carries no similarity signal, so Pinecone returns an
# arbitrary subset rather than a ranked one. Ask for as much as the index
# will give in one call; lowering this to 1000 once made a whole ranking
# report "no crop tags found" because the arbitrary 1000 happened to
# contain none.
AGGREGATION_TOP_K = 10_000

# Per-bucket cap on what reaches the model. Breadth-seeking intents get
# more, because "what are people talking about" answered from 12 records
# is a sample, not a theme.
BULLETS_PER_BUCKET = 12
BULLETS_PER_BUCKET_BROAD = 40
BROAD_INTENTS = ("topics",)

@dataclass
class SegmentEvidence:
    label: str
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)
    # Pre-truncation totals. Reporting the truncated count as if it were
    # the total put "Negative: 12" into an exported deck for a month that
    # actually had 60.
    totals: dict = field(default_factory=lambda: {"positive": 0, "negative": 0, "neutral": 0})

    @property
    def shown(self) -> int:
        return len(self.positive) + len(self.negative) + len(self.neutral)

    @property
    def total(self) -> int:
        return sum(self.totals.values())


@dataclass
class Evidence:
    segments: list[SegmentEvidence] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)   # raw, for counting modes
    complete: bool = True                              # False if the fetch was capped

    @property
    def shown(self) -> int:
        return sum(s.shown for s in self.segments)

    @property
    def total(self) -> int:
        return sum(s.total for s in self.segments)


def connect(api_key: str):
    """Returns (client, index): the client embeds, the index queries."""
    # Imported lazily so this module stays importable (and testable) without
    # the SDK, and so a cold start pays for it only when a query arrives.
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    return pc, pc.Index(PINECONE_INDEX_NAME)


def dataset_extent(index) -> tuple[str, str] | None:
    """Newest (month, year) present. Read from a stats record written at
    ingestion where possible; the fallback samples, which is why the old
    top_k=10 version could decide the 'latest year' from ten arbitrary
    records and scope a whole answer to the wrong year."""
    try:
        got = index.fetch(ids=[INDEX_STATS_ID])
        vectors = getattr(got, "vectors", None) or (got or {}).get("vectors", {})
        meta = (vectors.get(INDEX_STATS_ID) or {}).get("metadata") or {}
        if meta.get("max_month") and meta.get("max_year"):
            return meta["max_month"], str(meta["max_year"])
    except Exception:
        pass

    from vog.catalog import MONTH_ORDER
    try:
        res = index.query(vector=[0.0] * EMBEDDING_DIMENSION, top_k=1000, include_metadata=True)
        pairs = []
        for m in res.get("matches", []):
            md = m.get("metadata") or {}
            month, year = md.get("month"), md.get("year")
            if month in MONTH_ORDER and str(year).isdigit():
                pairs.append((int(year), month))
        if pairs:
            year, month = max(pairs, key=lambda p: (p[0], MONTH_ORDER[p[1]]))
            return month, str(year)
    except Exception:
        pass
    return None


def _filter_for(plan: QueryPlan, segment: Segment, month: str | None, year: str | None) -> dict:
    f = {}
    if month:
        f["month"] = {"$eq": month}
    if year:
        f["year"] = {"$eq": year}
    if segment.timeframe.week:
        week_num = P._week_number(segment.timeframe.week) if segment.timeframe.week else None
        if week_num:
            # Week is a database-level filter now. Applying it in Python
            # after a top_k cut silently dropped most of a week's records
            # whenever the month held more rows than top_k.
            f["week_num"] = {"$eq": week_num}
    if plan.category_filter:
        f["category"] = {"$eq": plan.category_filter}
    elif plan.intent == "positive":
        f["sentiment"] = {"$eq": "positive"}
    elif plan.intent == "complaint":
        f["sentiment"] = {"$eq": "negative"}
    return f


def _search(index, vector, filters: dict, top_k: int) -> list[dict]:
    res = index.query(
        vector=vector, top_k=top_k, include_metadata=True,
        filter=filters or None,
    )
    return [m for m in res.get("matches", [])
            if not (m.get("metadata") or {}).get("is_stats_record")]


def _bucket(matches: list[dict]) -> tuple[list[str], list[str], list[str], dict]:
    """Split matches into sentiment buckets, deduped, keeping true totals."""
    buckets = {"positive": [], "negative": [], "neutral": []}
    seen = {"positive": set(), "negative": set(), "neutral": set()}
    for m in matches:
        md = m.get("metadata") or {}
        value = str(md.get("value", "")).strip()
        if not value or P.is_empty_cell(value):
            continue
        sentiment = md.get("sentiment", "neutral")
        if sentiment not in buckets:
            sentiment = "neutral"
        entry = f"{md.get('category', '')}: {value}"
        # Dedupe on the feedback text alone. Keying on "category: value"
        # let the same line filed under two negative categories survive
        # twice and inflate the count.
        key = value.strip().lower()
        if key in seen[sentiment]:
            continue
        seen[sentiment].add(key)
        buckets[sentiment].append(entry)
    totals = {k: len(v) for k, v in buckets.items()}
    return buckets["positive"], buckets["negative"], buckets["neutral"], totals


def gather(plan: QueryPlan, index, pc, groq_api_key: str | None = None) -> Evidence:
    """Retrieve for every segment in the plan."""
    if plan.mode in ("rank", "trend"):
        return _gather_for_counting(plan, index)

    cap = BULLETS_PER_BUCKET_BROAD if plan.intent in BROAD_INTENTS else BULLETS_PER_BUCKET
    evidence = Evidence()

    for segment in plan.segments:
        vector = _vector_for(plan, segment, pc)
        months = segment.timeframe.months or [(None, None)]

        # Retrieve per month, then interleave newest-first so truncation
        # keeps the window representative. Concatenating chronologically
        # and slicing meant a six-month question was answered entirely
        # from its oldest month.
        per_month_pos, per_month_neg, per_month_neu = [], [], []
        for month, year in months:
            matches = _search(index, vector, _filter_for(plan, segment, month, year), 300)
            pos, neg, neu, _ = _bucket(matches)
            if segment.product:
                pos, neg, neu = (P.filter_bullets_by_product(b, segment.product) for b in (pos, neg, neu))
            if segment.crop:
                pos, neg, neu = (P.filter_bullets_by_crop(b, segment.crop) for b in (pos, neg, neu))
            per_month_pos.append(pos)
            per_month_neg.append(neg)
            per_month_neu.append(neu)

        pos = P._interleave_by_recency(per_month_pos)
        neg = P._interleave_by_recency(per_month_neg)
        neu = P._interleave_by_recency(per_month_neu)
        totals = {"positive": len(pos), "negative": len(neg), "neutral": len(neu)}

        # Intent decides which buckets are relevant at all.
        if plan.intent == "complaint":
            pos = []
        elif plan.intent == "positive":
            neg, neu = [], []
        elif plan.intent == "suggestion":
            pos, neg = [], []

        evidence.segments.append(SegmentEvidence(
            label=segment.label,
            positive=pos[:cap], negative=neg[:cap], neutral=neu[:cap],
            totals=totals,
        ))
    return evidence


def _gather_for_counting(plan: QueryPlan, index) -> Evidence:
    """Ranking and trend need every tagged record, not a relevance sample."""
    filters = {}
    if plan.intent == "positive":
        filters["sentiment"] = {"$eq": "positive"}
    elif plan.intent == "complaint":
        filters["sentiment"] = {"$eq": "negative"}
    elif plan.category_filter:
        filters["category"] = {"$eq": plan.category_filter}

    # Ranking and trend used to return before any timeframe was resolved,
    # so "the trend for the past three years" charted all of history.
    months = plan.segments[0].timeframe.months if plan.segments else []
    if months:
        filters["month"] = {"$in": sorted({m for m, _ in months})}
        filters["year"] = {"$in": sorted({y for _, y in months})}

    # Deliberately no try/except: swallowing an error here would be
    # indistinguishable from "the dataset is empty", which is the worst
    # possible failure for a path whose whole purpose is exact counting.
    res = index.query(
        vector=[0.0] * EMBEDDING_DIMENSION, top_k=AGGREGATION_TOP_K,
        include_metadata=True, filter=filters or None,
    )
    raw = res.get("matches", [])
    matches = [m for m in raw if not (m.get("metadata") or {}).get("is_stats_record")]
    return Evidence(matches=matches, complete=len(raw) < AGGREGATION_TOP_K)


def _vector_for(plan: QueryPlan, segment: Segment, pc):
    """Query embedding, nudged toward the subject but never replaced by it.

    Replacing it discarded everything specific the user asked, so "is
    Isabion too expensive", "what packaging problems does Isabion have"
    and "how does Isabion do on wheat" all retrieved the same generic
    bullets.
    """
    try:
        texts = [plan.query]
        subject = " ".join(filter(None, [segment.crop, segment.product]))
        if subject:
            texts.append(f"grower feedback about {subject}")
        vectors = llm.embed(texts, pc, input_type="query")
        if len(vectors) == 2:
            return P._blend_vectors(vectors[0], vectors[1], P.SUBJECT_BLEND_WEIGHT)
        return vectors[0]
    except Exception:
        return [0.0] * EMBEDDING_DIMENSION
