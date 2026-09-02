"""Question -> QueryPlan.

The old `process_chat_query` was 581 lines that interleaved parsing,
routing, retrieval and prompt-building, with three near-duplicate
retrieval branches (single period / time comparison / relative window)
that each re-implemented the same filter-and-truncate logic slightly
differently. That is where the recency bug lived: one branch sliced a
chronologically-ordered list and silently answered "the last 6 months"
using only the oldest one.

Everything collapses into one idea: **a question resolves to a list of
Segments**, and every mode is then the same loop over them.

    plain question      -> 1 segment
    "Jan vs Feb"        -> 2 segments, differing by timeframe
    "Tilt vs Isabion"   -> 2 segments, differing by subject
    "last 6 months"     -> 1 segment whose timeframe spans 6 months

This module does no I/O. It takes the few facts that need the index
(what the newest month is) as arguments, so it stays a pure function and
is fully testable without Pinecone.
"""

from dataclasses import dataclass, field

from vog import parsing as P
from vog.catalog import (
    DEFAULT_TIMEFRAME_LABEL, PRODUCT_QUERY_CATEGORY, SALES_KEYWORDS,
    SUGGESTION_CATEGORY,
)

# Modes are mutually exclusive answer shapes.
MODE_ANSWER = "answer"      # retrieve evidence, let the model write prose
MODE_RANK = "rank"          # deterministic counting, no model needed
MODE_TREND = "trend"        # deterministic time series
MODE_REPLY = "reply"        # canned reply, no retrieval at all

INTENTS = ("complaint", "positive", "suggestion", "sentiment", "topics")


@dataclass
class Timeframe:
    """One or more (month, year) pairs plus an optional week."""
    months: list[tuple[str, str]] = field(default_factory=list)
    week: str | None = None
    label: str = DEFAULT_TIMEFRAME_LABEL

    @property
    def is_open(self) -> bool:
        """No date constraint at all — search the whole dataset."""
        return not self.months


@dataclass
class Segment:
    """One retrievable slice. Comparisons are just several of these."""
    label: str
    timeframe: Timeframe
    product: str | None = None
    crop: str | None = None


@dataclass
class QueryPlan:
    query: str
    mode: str = MODE_ANSWER
    intent: str = "sentiment"
    intent_explicit: bool = False
    segments: list[Segment] = field(default_factory=list)
    comparison_axis: str | None = None   # "time" | "subject" | None
    rank_dimension: str | None = None    # "crop" | "product"
    rank_top_n: int = 10
    category_filter: str | None = None
    sales_scoped: bool = False
    output_format: str | None = None
    list_format: bool = False
    products_only: bool = False
    avoid_repeat: str | None = None
    canned_reply: str | None = None
    blocked: bool = False

    @property
    def needs_assist(self) -> bool:
        """True when regex detection found nothing to go on, so a model
        classification is worth a round trip. Deliberately narrow: the
        assist only fills a vacuum, it never overrides a real detection."""
        # sales_scoped questions are already routed to Product Queries by
        # SALES_KEYWORDS. Asking the model anyway is how a pricing question
        # got re-labelled "Suggestions": the fallback guessed an intent and
        # silently overrode a correct route.
        if self.mode != MODE_ANSWER or self.intent_explicit or self.sales_scoped:
            return False
        seg = self.segments[0] if self.segments else None
        return not (seg and (seg.product or seg.crop))

    @property
    def is_comparison(self) -> bool:
        return len(self.segments) > 1

    @property
    def subject_label(self) -> str | None:
        if self.is_comparison:
            return None  # the subjects ARE the axis; see build_header
        seg = self.segments[0] if self.segments else None
        return P.build_subject_label(seg.product, seg.crop) if seg else None


def _detect_intent(q: str) -> tuple[str, bool]:
    """Returns (intent, was_explicit). Ordered so the more specific
    categories win over the general 'sentiment' catch-all."""
    from vog.catalog import (
        COMPLAINT_KEYWORDS, POSITIVE_KEYWORDS, SENTIMENT_KEYWORDS,
        SUGGESTION_KEYWORDS, TOPIC_KEYWORDS,
    )
    for keywords, intent in (
        (TOPIC_KEYWORDS, "topics"),
        (COMPLAINT_KEYWORDS, "complaint"),
        (POSITIVE_KEYWORDS, "positive"),
        (SUGGESTION_KEYWORDS, "suggestion"),
        (SENTIMENT_KEYWORDS, "sentiment"),
    ):
        if any(k in q for k in keywords):
            return intent, True
    return "sentiment", False


def _rank_top_n(q: str, default: int = 10) -> int:
    """Honour the N the user asked for. The old code parsed 'top 5' to
    decide it was a ranking question, then hardcoded 10 rows anyway."""
    import re
    m = re.search(r'\btop\s+(\d+)\b', q)
    if m:
        return max(1, min(50, int(m.group(1))))
    words = {"three": 3, "five": 5, "ten": 10, "twenty": 20}
    m = re.search(r'\btop\s+(three|five|ten|twenty)\b', q)
    return words[m.group(1)] if m else default


def build_plan(
    user_query: str,
    *,
    latest_month_year: tuple[str, str] | None = None,
    prior_context: dict | None = None,
    assist: dict | None = None,
) -> QueryPlan:
    """Pure: everything needing the index is passed in.

    `latest_month_year` anchors relative dates ("last quarter") to the
    newest data actually present rather than to today's date — the
    dataset routinely lags the calendar, and anchoring to "now" returns
    an empty window while real recent data sits one month back.
    """
    q = user_query.lower()
    plan = QueryPlan(query=user_query)

    # ── Questions that are not data questions at all ──
    if P.detect_correction_or_meta_feedback(q):
        plan.mode, plan.canned_reply = MODE_REPLY, P.CORRECTION_ACK_REPLY
        return plan
    if P.detect_capability_question(q):
        plan.mode, plan.canned_reply = MODE_REPLY, P.CAPABILITY_REPLY
        return plan

    # Topic guardrail. A continuation phrase ("what about last month?") has
    # no domain word of its own but is plainly continuing an in-scope turn,
    # so it passes exactly when there is prior context for it to continue.
    if not P.is_query_in_scope(user_query) and not (
            prior_context and P.detect_followup_reference(q)):
        plan.mode, plan.blocked = MODE_REPLY, True
        plan.canned_reply = (
            "I cannot generate this response. I am strictly locked to analyzed "
            "dataset metrics and cannot find relevant information for this query."
        )
        return plan

    # ── Intent ──
    plan.intent, plan.intent_explicit = _detect_intent(q)

    if plan.intent == "suggestion":
        plan.category_filter = SUGGESTION_CATEGORY
    elif plan.intent == "sentiment" and any(k in q for k in SALES_KEYWORDS):
        # A pricing/availability question is a Product Queries question,
        # not a sentiment reading.
        plan.category_filter = PRODUCT_QUERY_CATEGORY
        plan.sales_scoped = True

    # ── Output shape ──
    plan.output_format = P.detect_output_format(q)
    plan.list_format = bool(P.detect_output_format and P._explicit_list_format(q))
    plan.products_only = P._wants_products_only(q)

    # ── Subjects ──
    products = P.detect_all_products(q)
    crops = P.detect_all_crops(q)
    product = products[0] if products else None
    crop = crops[0] if crops else None
    if product and crop and product.lower() == crop.lower():
        product, products = None, []

    # A model-proposed product/crop/intent for a question the regexes could
    # not read. Every value was validated against the real catalogs before
    # it got here, so the model can only ever hit an existing name, never
    # introduce one. It fills gaps and never overrides a real detection.
    if assist and not (product or crop) and not plan.intent_explicit:
        product = assist.get("product") or None
        crop = assist.get("crop") or None
        if assist.get("intent"):
            plan.intent, plan.intent_explicit = assist["intent"], True
            if plan.intent == "suggestion":
                plan.category_filter = SUGGESTION_CATEGORY

    # Follow-up inheritance: fills gaps only, never overrides, and only on
    # an explicit continuation phrase so an unrelated question is never
    # silently scoped by a previous turn.
    if prior_context and P.detect_followup_reference(q):
        product = product or prior_context.get("product")
        crop = crop or prior_context.get("crop")
        if not plan.intent_explicit and prior_context.get("intent"):
            plan.intent = prior_context["intent"]
            plan.intent_explicit = True
            if plan.intent == "suggestion":
                plan.category_filter = SUGGESTION_CATEGORY
        if P.detect_wants_more(q):
            plan.avoid_repeat = prior_context.get("last_reply")

    # ── Timeframe ──
    months = P.extract_all_months(q)
    years = P.extract_all_years(q)
    weeks = P.extract_all_weeks(q)
    week = weeks[0] if weeks else None

    # ── Mode ──
    plan.rank_dimension = P.detect_aggregation_request(q)
    if plan.rank_dimension:
        plan.mode = MODE_RANK
        plan.rank_top_n = _rank_top_n(q)
    elif P.detect_trend_request(q):
        plan.mode = MODE_TREND

    # ── Segments ──
    plan.segments, plan.comparison_axis = _build_segments(
        q, months, years, weeks, week, products, crops, product, crop,
        latest_month_year,
    )
    return plan


def _build_segments(q, months, years, weeks, week, products, crops,
                    product, crop, latest_month_year):
    """One list of segments, whichever axis the question compares on.

    Priority: an explicit multi-period question is a TIME comparison even
    if it also names two products; otherwise two subjects make a SUBJECT
    comparison; otherwise it is a single segment.
    """
    # Time comparison
    time_frames = _comparison_timeframes(months, years, weeks, latest_month_year)
    if time_frames:
        return [Segment(tf.label, tf, product, crop) for tf in time_frames], "time"

    # Subject comparison
    if len(products) >= 2:
        tf = _single_timeframe(q, months, years, week, latest_month_year)
        return [Segment(p.title(), tf, p, crop) for p in products], "subject"
    if len(crops) >= 2:
        tf = _single_timeframe(q, months, years, week, latest_month_year)
        return [Segment(P.canonical_crop(c), tf, product, c) for c in crops], "subject"

    tf = _single_timeframe(q, months, years, week, latest_month_year)
    return [Segment(tf.label, tf, product, crop)], None


def _comparison_timeframes(months, years, weeks, latest) -> list[Timeframe]:
    """Two or more explicit periods -> one Timeframe each."""
    fallback_year = latest[1] if latest else None
    if len(years) >= 2:
        month = months[0] if months else None
        week = weeks[0] if weeks else None
        return [
            Timeframe([(month, y)] if month else _all_months_of(y), week,
                      f"{month + ' ' if month else ''}{y}")
            for y in years
        ]
    if len(months) >= 2:
        year = years[0] if years else fallback_year
        week = weeks[0] if weeks else None
        if not year:
            return []
        return [Timeframe([(m, year)], week, f"{m} {year}") for m in months]
    if len(weeks) >= 2:
        month = months[0] if months else None
        year = years[0] if years else fallback_year
        if not (month and year):
            return []
        return [Timeframe([(month, year)], w, f"{w} week of {month} {year}") for w in weeks]
    return []


def _all_months_of(year: str) -> list[tuple[str, str]]:
    from vog.catalog import MONTH_ORDER
    return [(m, year) for m in MONTH_ORDER]


def _single_timeframe(q, months, years, week, latest) -> Timeframe:
    """One period: explicit month/year, a relative window, or open."""
    if months and years:
        return Timeframe([(months[0], years[0])], week, f"{months[0]} {years[0]}")
    if months and latest:
        # Month named without a year — assume the newest year present.
        return Timeframe([(months[0], latest[1])], week, f"{months[0]} {latest[1]}")
    if years:
        return Timeframe(_all_months_of(years[0]), week, years[0])

    window = P.detect_relative_window(q)
    if window and latest:
        resolved = _resolve_window(window, latest)
        if resolved:
            return resolved

    return Timeframe([], week, DEFAULT_TIMEFRAME_LABEL)


def _resolve_window(window: dict, latest: tuple[str, str]) -> Timeframe | None:
    """Relative window -> concrete months, anchored to the newest data."""
    latest_month, latest_year = latest
    idx = P._month_idx(latest_month, latest_year)
    kind = window["kind"]

    if kind == "last_n_months":
        n = max(1, window["n"])
        idxs, label = range(idx - n + 1, idx + 1), f"the last {n} month{'s' if n != 1 else ''}"
    elif kind == "this_month":
        idxs, label = [idx], f"{latest_month} {latest_year}"
    elif kind == "last_quarter":
        start = (idx // 3) * 3 - 3
        idxs, label = range(start, start + 3), "last quarter"
    elif kind == "this_quarter":
        idxs, label = range((idx // 3) * 3, idx + 1), "this quarter"
    elif kind == "since_month":
        target = P._month_idx(window["month"], latest_year)
        if target > idx:
            target -= 12
        idxs, label = range(target, idx + 1), f"since {window['month']}"
    else:
        return None

    return Timeframe([P._idx_to_month_year(i) for i in idxs], None, label)
