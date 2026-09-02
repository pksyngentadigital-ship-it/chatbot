"""Pure parsing, tagging and arithmetic.

Everything here is a function of its arguments — no Pinecone, no Groq, no
pandas. That is what makes the bulk of the system testable without a
network, and what keeps the serverless bundle small.
"""

import hashlib
import re
from collections import Counter

from vog.catalog import *  # noqa: F401,F403  (static data only)
from vog.catalog import (
    ALLOWED_GUARDRAIL_KEYWORDS, AMBIGUOUS_PRODUCT_WORDS, CATEGORY_NORMALIZE,
    CROP_CANONICAL, CROP_GUARDED, CROP_LIST, DEFAULT_TIMEFRAME_LABEL,
    DISEASE_PEST_WORDS, EMPTY_VALUES, GENERIC_CAPITALIZED_STOPWORDS,
    MAX_VALUE_CHARS, MONTH_MAP, MONTH_ORDER, MONTH_ORDER_INV, MONTH_TYPO_FIX,
    NEGATIVE_CATEGORIES, POSITIVE_CATEGORIES, PRODUCT_LIST, PRODUCT_QUERY_CATEGORY,
    PRODUCT_STOPWORDS, _MONTH_RE, _UPPERCASE_PRODUCT_TOKENS,
)

def normalize_category(raw_val):
    """Returns the canonical category name, or None when the value does not
    map to a known category. Previously an unknown spelling was passed
    through verbatim and then silently tagged sentiment='neutral' — so
    'Complaint / Negative Feedback' (spaces around the slash) made an entire
    sheet of complaints invisible to every complaint query. Callers now get
    None and are expected to record it as a skipped/unmapped value."""
    if raw_val is None:
        return None
    cleaned = re.sub(r'\s*/\s*', '/', re.sub(r'\s+', ' ', str(raw_val)).strip().lower())
    if not cleaned or cleaned in EMPTY_VALUES:
        return None
    if cleaned in CATEGORY_NORMALIZE:
        return CATEGORY_NORMALIZE[cleaned]
    # Tolerate a trailing plural ("Positive Feedbacks").
    if cleaned.endswith("s") and cleaned[:-1] in CATEGORY_NORMALIZE:
        return CATEGORY_NORMALIZE[cleaned[:-1]]
    return None

def is_empty_cell(value: str) -> bool:
    return value.strip().lower() in EMPTY_VALUES or value.strip() == ""

def split_bullets(cell_text: str) -> list[str]:
    """Split a cell into individual feedback points. Splits on newlines AND
    on inline bullet glyphs / sentence-ending semicolons — a cell authored
    as "• A • B • C" on one line used to become a single record with one
    blended set of crop/product tags."""
    parts = re.split(r'[\n\r]+|\s+[•●·]\s*|(?<=[.;])\s+(?=[A-Z0-9])', cell_text)
    bullets = []
    for line in parts:
        if line is None:
            continue
        clean = re.sub(r'^[\s•●·\-–—]+', '', line).strip()
        if clean and clean.lower() not in EMPTY_VALUES and len(clean) >= 2:
            bullets.append(clean)
    return bullets

def extract_month_from_col(col: str) -> str:
    """Word-boundary month match. Without \\b the alternation matched inside
    ordinary words — 'Remarks' -> March, 'Summary' -> March,
    'Decrease' -> December, 'Separate' -> September — which filed real rows
    under a month nobody wrote."""
    fixed = str(col)
    for typo, correct in MONTH_TYPO_FIX.items():
        fixed = re.sub(r'\b' + typo + r'\b', correct, fixed, flags=re.IGNORECASE)
    match = _MONTH_RE.search(fixed)
    if not match:
        return "Unknown"
    return MONTH_MAP.get(match.group(1).lower(), match.group(1).capitalize())

def find_category_column(df_columns):
    """Prefer an exact 'category' header. Falling back to the first header
    merely CONTAINING 'categ' picked 'Sub Category' over 'Category' and
    'Categorization Notes' over the real column — and the wrong column then
    drove the entire Layout A branch, tagging every record with a value that
    normalizes to nothing and therefore lands as sentiment='neutral'."""
    exact_targets = {"category", "case category", "casecategory"}
    for col in df_columns:
        if re.sub(r'\s+', ' ', str(col)).strip().lower() in exact_targets:
            return col
    for col in df_columns:
        if 'categ' in re.sub(r'\s+', '', str(col)).lower():
            return col
    return None

def infer_year_for_sheet(sheet_name: str, all_sheet_names: list) -> str | None:
    """ Sheets carrying an explicit 4-digit year are trusted directly. Undated legacy sheets ("Jan till June") are dated from their NEAREST dated neighbour by position — the previous implementation took the global min of all later sheets / max of all earlier ones, which for ['Legacy', 'VOG 2026', 'VOG 2025'] produced 2024 instead of 2025, and could silently assign a legacy sheet the same year as a real dated sheet, double-counting every record in it. Returns None rather than guessing when the inferred year would collide with an explicitly-dated sheet. """
    direct = re.search(r'\b(20\d{2})\b', sheet_name.strip())
    if direct:
        return direct.group(1)

    dated = [
        (i, int(m.group(1)))
        for i, name in enumerate(all_sheet_names)
        for m in [re.search(r'\b(20\d{2})\b', str(name).strip())] if m
    ]
    if not dated:
        return None

    my_index = None
    for i, name in enumerate(all_sheet_names):
        if str(name).strip() == sheet_name.strip():
            my_index = i
            break
    if my_index is None:
        return None

    # Nearest dated sheet by positional distance, not global min/max.
    nearest_i, nearest_year = min(dated, key=lambda t: (abs(t[0] - my_index), t[0]))
    guess = nearest_year - 1 if nearest_i > my_index else nearest_year + 1

    claimed = {year for _, year in dated}
    if guess in claimed:
        # Refuse rather than duplicate an explicitly-dated sheet's year.
        return None
    return str(guess)

def canonical_crop(crop: str) -> str:
    """Collapse crop synonyms (paddy->Rice, corn->Maize) to one label."""
    return CROP_CANONICAL.get(crop.strip().lower(), crop.strip().title())

def extract_crops(text: str) -> list[str]:
    """Tag every known crop mentioned in a feedback bullet, for ingestion-time
    metadata. Synonyms are collapsed so each real crop is counted once."""
    text_lower = text.lower()
    found = []
    for crop in CROP_LIST:
        if re.search(r'\b' + re.escape(crop) + r'\b', text_lower):
            label = canonical_crop(crop)
            if label not in found:
                found.append(label)
    for pattern, label in CROP_GUARDED.items():
        if re.search(pattern, text_lower) and label not in found:
            found.append(label)
    return found

def _canonical_product(product: str) -> str:
    return " ".join(
        w.upper() if w.lower() in _UPPERCASE_PRODUCT_TOKENS else w.capitalize()
        for w in product.split()
    )

def extract_product_mentions(text: str) -> list[str]:
    """Product tags: matches against the curated catalog, and nothing else.

    There is no guessing step. A capitalized-phrase heuristic used to run
    alongside this and write into the same tag; it cannot tell a brand from
    any other capitalized noun, and in production it filled the ranking
    with agronomic vocabulary — 12 of the top 20 "products" were things
    like Abiotic, Early (from "Early Blight"), Blossom, White (from "White
    Fly") and Potash (a fragment of "Naya Potash").

    With PRODUCT_LIST now derived from the official price list, the catalog
    IS the source of truth: a new product enters the system by being added
    to the price list and re-ingested, not by being inferred from text.
    """
    text_lower = text.lower()

    catalog_hits = []
    for product in PRODUCT_LIST:
        if not re.search(r'\b' + re.escape(product) + r'\b', text_lower):
            continue
        if product in AMBIGUOUS_PRODUCT_WORDS:
            # Require the capitalized form in the original text.
            if not re.search(r'\b' + re.escape(product.title()) + r'\b', text):
                continue
        catalog_hits.append(product)

    # Drop any catalog hit that is a strict prefix of a longer hit, so
    # "Isabion Gold" does not also credit "Isabion".
    filtered = [
        p for p in catalog_hits
        if not any(other != p and other.startswith(p + " ") for other in catalog_hits)
    ]

    out, seen = [], set()
    for product in filtered:
        canonical = _canonical_product(product)
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            out.append(canonical)
    return out

def _mentions(text: str, term: str) -> bool:
    """Word-boundary containment. Bare substring matching made 'rice' match
    'price' and 'gram' match 'program' — and it was inconsistent with
    extract_crops, which has always used \\b at ingestion time."""
    return bool(re.search(r'\b' + re.escape(term.lower()) + r'\b', text.lower()))

def _week_number(week_label: str) -> int | None:
    """Parse the ordinal from a week label ('2nd Week January' -> 2).

    Matches the ordinal specifically rather than the first integer in the
    string — a column named 'March 2026 Week 1' previously yielded 2026.
    """
    m = re.search(r'\b(\d{1,2})\s*(?:st|nd|rd|th)\b', week_label, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 6 else None
    m = re.search(r'\bweek\s*(\d{1,2})\b', week_label, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 6 else None
    return None

def _clean_cell_text(cell_val) -> str | None:
    """Return usable feedback text, or None if this cell isn't feedback.

    Guards against non-text cells: a date cell stringified to
    '2026-01-05 00:00:00' and a float to '12.0', both of which were being
    embedded and counted as grower feedback.
    """
    if cell_val is None:
        return None
    if isinstance(cell_val, (int, float, bool)):
        return None
    text = str(cell_val).strip()
    if not text or is_empty_cell(text):
        return None
    if re.fullmatch(r'[\d.\-:/ ]+', text):
        return None
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}[ T].*', text):
        return None
    return text

def _vector_id(sheet: str, category: str, week_label: str, row: int, bullet_idx: int, bullet: str) -> str:
    """Content-addressed id.

    The previous scheme concatenated row and bullet indexes with no
    separator ("...{idx}{b_idx}"), so row 1/bullet 12 and row 11/bullet 2
    both produced "...112" and the second silently overwrote the first —
    16 records reported ingested, 13 actually stored. Hashing the content
    also makes re-ingesting unchanged rows idempotent.
    """
    basis = f"{sheet}|{category}|{week_label}|{row}|{bullet_idx}|{bullet}"
    return "v_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()

def _make_metadata_payload(inferred_year, row_month, week_label, category, bullet,
                           sheet_name="", src_row=-1, ingest_run="", week_num=None):
    """Build the embedding text and the Pinecone metadata for one bullet.

    Provenance (sheet / src_row / ingest_run) is stored so a re-ingest can
    delete exactly what it is replacing. Without it there was no way to
    scope a delete, so corrections could never take effect: the old vector
    simply stayed in the index forever.

    The old "text" field duplicated the whole bullet inside a longer
    sentence and was stored alongside "value", roughly doubling metadata
    size for no benefit — a single long cell could push a payload past
    Pinecone's 40KB limit and reject the entire 50-vector batch. It is
    reconstructible from the other fields, so it is no longer stored.
    """
    is_positive = category in POSITIVE_CATEGORIES
    is_negative = category in NEGATIVE_CATEGORIES
    context_chunk = (
        f"Year: {inferred_year}. "
        f"Month: {row_month}. "
        f"Week: {week_label}. "
        f"Case Category: {category}. "
        f"Feedback: {bullet}."
    )
    metadata = {
        "month":     row_month,
        "year":      inferred_year,
        "week":      week_label,
        "category":  category,
        "sentiment": (
            "positive" if is_positive
            else "negative" if is_negative
            else "neutral"
        ),
        "value":    bullet[:MAX_VALUE_CHARS],
        "crop":     ",".join(extract_crops(bullet)),
        # Catalog matches only — the price list is the source of truth.
        "products": ",".join(extract_product_mentions(bullet)),
        "sheet":    sheet_name,
        "src_row":  int(src_row),
        "ingest_run": ingest_run,
    }
    if week_num is not None:
        # Stored as an integer so week can be a database-level filter
        # instead of a Python substring test applied after the top_k cut.
        metadata["week_num"] = int(week_num)
    if len(bullet) > MAX_VALUE_CHARS:
        metadata["truncated"] = True
    return context_chunk, metadata

def extract_all_months(query_lower: str) -> list[str]:
    """Every distinct month named, in the order the user actually wrote them.

    Two fixes over the naive version:

    * It iterated the month vocabulary longest-first and appended in THAT
      order, so "compare march and january" came back ['January','March']
      — the docstring said "first seen" but the output was sorted by
      keyword length. On a two-period comparison this silently reversed
      the periods in the answer.
    * Bare "may" is far more often the modal verb than the month.
      "what suggestions may improve availability" was being scoped to
      May. It now only counts as a month next to a year, an adjacent
      month, or a date-ish preposition.
    """
    hits: list[tuple[int, str]] = []
    for token in sorted(MONTH_MAP.keys(), key=len, reverse=True):
        for m in re.finditer(r'\b' + re.escape(token) + r'\b', query_lower):
            if token == "may" and not _may_is_a_month(query_lower, m.start(), m.end()):
                continue
            # Skip a short form already covered by a longer match here
            # ("jan" inside "january").
            if any(s <= m.start() < e for s, e in
                   [(h[0], h[0] + len(h[1])) for h in hits]):
                continue
            hits.append((m.start(), MONTH_MAP[token]))

    out: list[str] = []
    for _, month in sorted(hits, key=lambda t: t[0]):
        if month not in out:
            out.append(month)
    return out


_MAY_MONTH_CONTEXT = re.compile(
    r'(?:\bin\s+may\b|\bof\s+may\b|\bmay\s+20\d{2}\b|\bsince\s+may\b|'
    r'\bmay\s+and\b|\band\s+may\b|\bmay\s+to\b|\bto\s+may\b|'
    r'\b(?:1st|2nd|3rd|4th|5th)\s+week\s+(?:of\s+)?may\b)'
)


def _may_is_a_month(query_lower: str, start: int, end: int) -> bool:
    """"May" counts as the month only in unambiguously date-like phrasing."""
    return bool(_MAY_MONTH_CONTEXT.search(query_lower))

def extract_all_years(query_lower: str) -> list[str]:
    """Return every distinct 4-digit year mentioned, in order first seen."""
    return list(dict.fromkeys(re.findall(r'\b(20\d{2})\b', query_lower)))

def extract_all_weeks(query_lower: str) -> list[str]:
    """Return every distinct 'Nth week' phrase mentioned, normalized."""
    ORDINAL_MAP = {
        "first": "1st", "second": "2nd",
        "third": "3rd", "fourth": "4th", "fifth": "5th"
    }
    raw_matches = re.findall(
        r'\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\s+week\b',
        query_lower
    )
    normalized = []
    for m in raw_matches:
        val = ORDINAL_MAP.get(m, m)
        if val not in normalized:
            normalized.append(val)
    return normalized

def detect_all_products(query_lower: str) -> list[str]:
    """Every catalog product named in the query, in the order the USER wrote
    them, longest-match-first so "Isabion Gold" wins over "Isabion".

    detect_product_known returns only the first match and scans in CATALOG
    order, so "compare Tilt and Isabion" silently answered about Isabion
    alone — and not even because it came first in the question, but because
    it sits earlier in PRODUCT_LIST.
    """
    hits = []
    for product in sorted(PRODUCT_LIST, key=len, reverse=True):
        m = re.search(r'\b' + re.escape(product) + r'\b', query_lower)
        if m:
            # Skip a shorter product fully contained in one already matched.
            if any(product in longer and product != longer for longer, _ in hits):
                continue
            hits.append((product, m.start()))
    return [p for p, _ in sorted(hits, key=lambda t: t[1])]

def detect_all_crops(query_lower: str) -> list[str]:
    """Every catalog crop named in the query, in the order the user wrote them."""
    hits = []
    for crop in sorted(CROP_LIST, key=len, reverse=True):
        m = re.search(r'\b' + re.escape(crop) + r'\b', query_lower)
        if m:
            if any(crop in longer and crop != longer for longer, _ in hits):
                continue
            hits.append((crop, m.start()))
    ordered = [c for c, _ in sorted(hits, key=lambda t: t[1])]
    # Collapse synonyms so "rice and paddy" isn't treated as two subjects.
    seen, out = set(), []
    for c in ordered:
        key = canonical_crop(c).lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

def detect_product_known(query_lower: str) -> str | None:
    """Fast path: the first catalog product named in the query."""
    products = detect_all_products(query_lower)
    return products[0] if products else None

def detect_crop(query_lower: str) -> str | None:
    """Fast path: match against the curated CROP_LIST (closed vocabulary, no dynamic probe needed)."""
    for crop in CROP_LIST:
        if re.search(r'\b' + re.escape(crop) + r'\b', query_lower):
            return crop
    return None

def filter_bullets_by_product(bullets: list[str], product: str) -> list[str]:
    """Keep only bullets that actually reference the requested product."""
    return [b for b in bullets if _mentions(b, product)]

def filter_bullets_by_crop(bullets: list[str], crop: str) -> list[str]:
    """Keep only bullets that actually reference the requested crop, allowing
    for synonyms (a 'rice' query should also match bullets saying 'paddy')."""
    aliases = {crop.lower()} | {
        k for k, v in CROP_CANONICAL.items() if v.lower() == canonical_crop(crop).lower()
    } | {canonical_crop(crop).lower()}
    return [b for b in bullets if any(_mentions(b, a) for a in aliases)]

def _month_idx(month: str, year: str) -> int:
    """Absolute month index (months since year 0) — makes month arithmetic a plain integer add/subtract instead of manual year-rollover juggling."""
    return int(year) * 12 + (MONTH_ORDER[month] - 1)

def _idx_to_month_year(idx: int) -> tuple[str, str]:
    year, month0 = divmod(idx, 12)
    return MONTH_ORDER_INV[month0 + 1], str(year)

def detect_relative_window(query_lower: str) -> dict | None:
    """ Detects relative time-window phrases that span MULTIPLE months — "last 30 days", "last quarter", "last 3 months", "since March" — which the exact single month/year/week matchers can't express at all today. Returns a small descriptor dict for resolve_relative_window(), or None. Note on granularity: the ingested data only carries month/week tags, no exact calendar date — so "last N days" and "last N weeks" are necessarily approximated to the nearest whole month count (round(N/30) and round(N/4) respectively). That's a real, documented limitation of the data model, not a rounding bug. """
    m = re.search(r'\b(?:last|past)\s+(\d+)\s+days?\b', query_lower)
    if m:
        months_back = max(1, round(int(m.group(1)) / 30))
        return {"kind": "last_n_months", "n": months_back}

    m = re.search(r'\b(?:last|past)\s+(\d+)\s+weeks?\b', query_lower)
    if m:
        months_back = max(1, round(int(m.group(1)) / 4))
        return {"kind": "last_n_months", "n": months_back}

    m = re.search(r'\b(?:last|past|previous)\s+(\d+)\s+months?\b', query_lower)
    if m:
        return {"kind": "last_n_months", "n": int(m.group(1))}

    if re.search(r'\blast\s+month\b', query_lower):
        return {"kind": "last_n_months", "n": 1}

    if re.search(r'\bthis\s+month\b|\bcurrent\s+month\b', query_lower):
        return {"kind": "this_month"}

    if re.search(r'\b(?:last|past|previous)\s+quarter\b', query_lower):
        return {"kind": "last_quarter"}

    if re.search(r'\bthis\s+quarter\b|\bcurrent\s+quarter\b', query_lower):
        return {"kind": "this_quarter"}

    m = re.search(
        r'\bsince\s+(january|february|march|april|may|june|july|august'
        r'|september|october|november|december'
        r'|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b',
        query_lower
    )
    if m:
        return {"kind": "since_month", "month": MONTH_MAP.get(m.group(1), m.group(1).capitalize())}

    return None

def _interleave_by_recency(per_month_bullets: list[list[str]]) -> list[str]:
    """ Merge per-month bullet lists (given oldest-month-first) into one list that stays representative after downstream truncation. Round-robins one bullet from each month starting with the NEWEST, so a "last N months" answer covers the whole window with a recency bias, instead of being filled entirely by whichever month happens to come first. Deduplicates case-insensitively. """
    merged, seen = [], set()
    newest_first = list(reversed(per_month_bullets))
    for i in range(max((len(b) for b in newest_first), default=0)):
        for month_bullets in newest_first:
            if i < len(month_bullets):
                b = month_bullets[i]
                key = b.strip().lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(b)
    return merged

def _blend_vectors(query_vec, subject_vec, subject_weight: float):
    """Weighted blend of two embeddings, re-normalized to unit length.

    Re-normalizing matters: cosine similarity is scale-invariant but some
    index configurations use dot product, where an un-normalized blend
    would silently change the score magnitude.
    """
    if not query_vec or not subject_vec or len(query_vec) != len(subject_vec):
        return query_vec
    w = max(0.0, min(1.0, subject_weight))
    blended = [(1.0 - w) * q + w * s for q, s in zip(query_vec, subject_vec)]
    norm = sum(v * v for v in blended) ** 0.5
    if norm == 0:
        return query_vec
    return [v / norm for v in blended]

# How much the detected subject pulls the retrieval vector. Below ~0.5 the
# subject barely lifts on-topic records; above it, the question's own terms
# stop mattering — which is the behaviour being fixed.
SUBJECT_BLEND_WEIGHT = 0.45

def rank_by_field(matches, field: str, top_n: int = 10):
    """Count occurrences of each comma-separated tag value in the given metadata field, most common first."""
    counter = Counter()
    for m in matches:
        raw = m.get("metadata", {}).get(field, "")
        if not raw:
            continue
        for val in str(raw).split(","):
            val = val.strip()
            if val:
                counter[val] += 1
    return counter.most_common(top_n)

def compute_monthly_trend(matches):
    """ Group raw Pinecone matches by (year, month) and return a chronologically sorted list of (label, count) tuples, e.g. [("January 2026", 42), ("February 2026", 51), ...]. Purely deterministic counting — no LLM involved, so the numbers are always exactly what's in the data. """
    counts = Counter()
    for m in matches:
        md = m.get("metadata", {})
        month = md.get("month")
        year = md.get("year")
        if not month or not year or month not in MONTH_ORDER:
            continue
        counts[(year, month)] += 1

    ordered_keys = sorted(counts.keys(), key=lambda k: (int(k[0]), MONTH_ORDER[k[1]]))
    return [(f"{month} {year}", counts[(year, month)]) for year, month in ordered_keys]

def densify_monthly_counts(monthly_counts):
    """Insert zero-count entries for months with no records.

    compute_monthly_trend only emits months that have data, so walking the
    result by index treats non-adjacent months as consecutive — a series of
    [Nov 2025, Feb 2026] produced a "+300% month-over-month" label across a
    three-month gap.
    """
    if not monthly_counts:
        return []
    parsed = []
    for label, count in monthly_counts:
        month, year = label.rsplit(" ", 1)
        parsed.append((_month_idx(month, year), count))
    out = []
    for idx in range(parsed[0][0], parsed[-1][0] + 1):
        month, year = _idx_to_month_year(idx)
        match = next((c for i, c in parsed if i == idx), 0)
        out.append((f"{month} {year}", match))
    return out


def compute_growth_series(monthly_counts):
    """ Given [(label, count), ...] in chronological order, return a parallel list of month-over-month growth percentages (None for the first period, which has no prior month to compare against). """
    growth = [None]
    for i in range(1, len(monthly_counts)):
        prev_count = monthly_counts[i - 1][1]
        curr_count = monthly_counts[i][1]
        if prev_count == 0:
            growth.append(None)
        else:
            growth.append(round((curr_count - prev_count) / prev_count * 100, 1))
    return growth

def split_into_points(text: str, max_points: int = 6) -> list[str]:
    """Break an LLM prose/bullet response into short standalone points for slide bullets."""
    lines = [l.strip(" -*").strip() for l in text.split("\n") if l.strip(" -*").strip()]
    if len(lines) >= 2:
        return lines[:max_points]
    # Fall back to sentence-splitting for single-paragraph prose responses.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()][:max_points]

def detect_aggregation_request(query_lower: str) -> str | None:
    """ Detects queries that want a deterministic, counted ranking ("which crop generated the highest number of complaints", "products with the highest complaint frequency") rather than free-form LLM summarization. Returns 'crop' or 'product' — the metadata field to rank by — or None. Kept separate from the normal retrieval flow because counting must be exact (computed in Python from real metadata tags), never left to the LLM to eyeball from a handful of retrieved bullets. Requires crop/product to be the explicit SUBJECT of the ranking (e.g. "which crop...", "products with the highest...") — a bare co-occurrence of a rank word and the word "product" isn't enough, since that also matches open-ended synthesis asks like "most common product improvement recommendations", which want an LLM to cluster free text, not a tag count. """
    synthesis_markers = [
        'recommend', 'suggestion', 'insight', 'improvement', 'expectation',
        'root cause', 'strategic action',
    ]
    if any(m in query_lower for m in synthesis_markers):
        return None

    crop_subject = bool(
        re.search(r'\bwhich\s+crops?\b', query_lower)
        or re.search(r'\bcrops?\s+(?:with|that|generated|received|has|have)\b', query_lower)
        or re.search(r'\btop\s+\d+\s+crops?\b', query_lower)
        or re.search(r'\bcrop[- ]wise\s+ranking\b', query_lower)
    )
    product_subject = bool(
        re.search(r'\bwhich\s+products?\b', query_lower)
        or re.search(r'\bproducts?\s+(?:with|that|generated|received|has|have)\b', query_lower)
        or re.search(r'\btop\s+\d+\s+products?\b', query_lower)
        or re.search(r'\bproduct[- ]wise\s+ranking\b', query_lower)
    )
    if not (crop_subject or product_subject):
        return None

    rank_phrases = [
        'highest number of', 'highest', 'most frequent', 'most complaints',
        'most common', 'ranking', 'rank ', 'frequency', 'greatest'
    ]
    wants_ranking = (
        any(p in query_lower for p in rank_phrases)
        or bool(re.search(r'\btop\s+\d+\b', query_lower))
        or bool(re.search(r'\btop\s+(five|ten|three)\b', query_lower))
    )
    if not wants_ranking:
        return None

    return 'crop' if crop_subject else 'product'

def detect_trend_request(query_lower: str) -> bool:
    """ Detects requests for a monthly trend / growth-over-time breakdown (as opposed to a single-period or comparison-of-two-periods question). Deliberately conservative — bare words like "sales" or "products" alone are far too common in ordinary questions to trigger a full trend workup, so this requires an explicit temporal/trend framing. """
    trend_phrases = [
        'trend', 'trends', 'trending', 'month over month', 'month-over-month',
        'monthly trend', 'over time', 'growth', 'analytics over time',
        'performance over', 'monthly totals', 'monthly breakdown',
        # "...by month" / "...per month" / "monthly ..." are the plainest way
        # to ask for a time series, and were missing: "show overall grower
        # sentiment by month" — one of the app's own suggested prompts — fell
        # through to a single-period summary, and the model then said the data
        # contained no monthly information.
        'by month', 'per month', 'each month', 'by week', 'per week',
        'month by month', 'monthly',
    ]
    return any(p in query_lower for p in trend_phrases)

def detect_output_format(query_lower: str) -> str | None:
    """Detects which presentation format the user explicitly asked for."""
    if re.search(r'\bexcel\b|\bexport\b|\bdownload\b', query_lower):
        return 'excel'
    if re.search(r'\bppt\b|\bpowerpoint\b|\bpresentation\b|\bslides?\b', query_lower):
        return 'ppt'
    if re.search(r'\bexecutive\s+summary\b|\bone[- ]page\b|\bexec\s+summary\b', query_lower):
        return 'exec_summary'
    if re.search(r'\btable\b', query_lower):
        return 'table'
    if re.search(r'\bchart\b|\bgraph\b|\bvisuali[sz]e?\b|\bvisuali[sz]ation\b', query_lower):
        return 'chart'
    return None

def detect_followup_reference(query_lower: str) -> bool:
    """ Detects explicit conversational continuation phrasing ("what about wheat?", "and for Isabion?") — deliberately narrow (exact phrase match, not just "short query") so an unrelated fresh question never accidentally inherits stale context from a few turns ago. """
    return any(p in query_lower for p in FOLLOWUP_PHRASES)

def detect_wants_more(query_lower: str) -> bool:
    """Detects a request for MORE/DIFFERENT points on the same subject, as opposed to a plain repeat of the previous question — used to tell the LLM what was already said so it doesn't repeat itself."""
    return any(p in query_lower for p in MORE_INSIGHTS_PHRASES)

def detect_capability_question(query_lower: str) -> bool:
    """Detects a user asking what the tool can do, rather than asking it for data."""
    return any(p in query_lower for p in CAPABILITY_PHRASES)

def detect_correction_or_meta_feedback(query_lower: str) -> bool:
    """Detects a message that's giving feedback/correcting the chatbot's own behavior or knowledge, rather than asking a data question — see CORRECTION_PHRASES for the coverage caveat."""
    return any(p in query_lower for p in CORRECTION_PHRASES)

def is_query_in_scope(user_query: str) -> bool:
    """Strict topic guardrail: at least one recognized domain word must appear."""
    query_words = re.findall(r'\b\w+\b', user_query.lower())
    return any(word in ALLOWED_GUARDRAIL_KEYWORDS for word in query_words)

# Continuation phrasings that carry the previous turn's subject forward.
# The original built this by reassigning itself after MORE_INSIGHTS_PHRASES
# was defined; stated once here instead.
CONTINUATION_PHRASES = [
    "what about", "how about", "what's about", "and what about",
    "same for", "also for", "and for", "what of", "and about",
    "what if", "and how about",
]

# Phrases that additionally ask for MORE/DIFFERENT points on the same
# subject, so the previous answer is passed back to the model as
# "already said" rather than being repeated.
MORE_INSIGHTS_PHRASES = [
    "what other", "any other", "anything else", "what else",
    "else can you", "more insight", "other insight", "something else",
    "what more", "anything more", "else you can", "more detail",
    "more information", "additional insight", "additional detail",
]

FOLLOWUP_PHRASES = CONTINUATION_PHRASES + MORE_INSIGHTS_PHRASES


# Phrases that signal the user is correcting or giving feedback ABOUT the
# chatbot itself ("Kaho is not a product", "you're wrong about that") —
# rather than asking a question about the grower-feedback data. Deliberately
# a deterministic phrase list, not an LLM call: this class of message is
# rare enough that adding LLM latency/cost to every single message just to
# catch it isn't worth it, mirroring the cheap-keyword-list-first approach
# used for intent detection. Real natural language can phrase a correction
# in effectively unlimited ways, so — like the relative-date parser — this
# is a real, documented coverage limit, not a claim of completeness.
CORRECTION_PHRASES = [
    "is not a product", "isn't a product", "not a real product",
    "is not a real product", "is not a crop", "isn't a crop",
    "not a real crop", "that's wrong", "that is wrong", "that's incorrect",
    "that is incorrect", "that's not correct", "that is not correct",
    "you're wrong", "you are wrong", "you're mistaken", "you are mistaken",
    "you made a mistake", "you made an error", "this is incorrect",
    "this is wrong", "please fix this", "please correct this",
    "stop treating", "stop showing", "stop labeling", "stop calling",
    "should not be treated as", "shouldn't be treated as",
    "you got that wrong", "you got this wrong",
]

CORRECTION_ACK_REPLY = (
    "Thanks for flagging that — I don't have a way to update my product/crop "
    "catalog or behavior directly from a chat message, since that's shared "
    "across everyone using this tool and changing it here could affect "
    "results for other users. I've logged this note for the team to review "
    "and correct in the underlying configuration if confirmed."
)

# "What can I ask you?" is the most natural opening question a new user
# has, and it contains no domain vocabulary — so the topic guardrail
# refused it with "I cannot generate this response", which reads as the
# tool being broken rather than as a scope boundary.
CAPABILITY_PHRASES = [
    "what can you do", "what can you generate", "what can you tell me",
    "what can i ask", "what should i ask", "what questions can",
    "what kind of questions", "what type of questions", "how do i use",
    "how does this work", "what are you", "who are you", "help me get started",
    "what are your capabilities", "capabilities", "what do you do",
    "give me some examples", "example questions", "sample questions",
    "what can this do", "what is this",
]

CAPABILITY_REPLY = (
    "I answer questions about your ingested grower-feedback data. Here's what I can do:\n\n"
    "**Sentiment & themes**\n"
    "- Overall sentiment, positive feedback, or complaints — for everything, or scoped to one product or crop\n"
    "- What growers are talking about most (themes, not just good/bad)\n"
    "- Suggestions and improvement requests growers have raised\n\n"
    "**Rankings & trends** *(counted exactly, never estimated)*\n"
    "- Which crop or product has the most complaints or positive feedback\n"
    "- Monthly trends with month-over-month change\n\n"
    "**Comparisons**\n"
    "- Two time periods — \"compare January 2026 and February 2026\"\n"
    "- Two products or crops — \"compare Tilt and Isabion\"\n\n"
    "**Time filters**\n"
    "- A specific month or year, \"the last 3 months\", \"last quarter\", \"since March\"\n\n"
    "**Exports** — every answer can be downloaded as CSV, Excel or PowerPoint.\n\n"
    "Try: *\"Which crop generated the highest number of complaints?\"* or "
    "*\"What are growers saying about Isabion?\"*\n\n"
    "One limit worth knowing: I only use your ingested feedback. I won't answer "
    "from general knowledge, and if the data doesn't cover something I'll say so."
)

def build_subject_label(active_product, active_crop):
    """Combine crop + product into one display label, e.g. 'Wheat + Isabion'. Collapses to one when both detections landed on the same word (e.g. dynamic product-probe fallback re-matching the crop name)."""
    if active_crop and active_product and active_crop.lower() == active_product.lower():
        return active_crop.title()
    parts = []
    if active_crop:
        parts.append(active_crop.title())
    if active_product:
        parts.append(active_product.title())
    return " + ".join(parts) if parts else None

def build_header(query_intent, timeframe_label, active_product, periods, active_crop=None, category_filter=None):
    """ Product/crop and comparison context always take priority over the generic 'period' heading — a product or crop query is labeled with its subject (never falls back to a generic 'sentiment overview for the period' heading), and a comparison query is clearly labeled as a comparison. """
    subject_label = build_subject_label(active_product, active_crop)

    if periods:
        period_join = " 🆚 ".join(p[0] for p in periods)
        subject = f"{subject_label} — " if subject_label else ""
        if query_intent == "complaint":
            return f"🔀 {subject}Complaints Comparison: {period_join}\n\n"
        elif query_intent == "positive":
            return f"🔀 {subject}Positive Feedback Comparison: {period_join}\n\n"
        elif query_intent == "suggestion":
            return f"🔀 {subject}Suggestions Comparison: {period_join}\n\n"
        else:
            return f"🔀 {subject}Sentiment Comparison: {period_join}\n\n"

    if subject_label:
        suffix = f" ({timeframe_label})" if timeframe_label != DEFAULT_TIMEFRAME_LABEL else ""
        if query_intent == "complaint":
            return f"🐛 Complaints about {subject_label}{suffix}:\n\n"
        elif query_intent == "positive":
            return f"🌻 Positive Feedback about {subject_label}{suffix}:\n\n"
        elif query_intent == "suggestion":
            return f"💡 Suggestions about {subject_label}{suffix}:\n\n"
        elif query_intent == "topics":
            return f"🗣️ What's Being Said About {subject_label}{suffix}:\n\n"
        else:
            return f"🌾 {subject_label} — Sentiment Overview{suffix}:\n\n"

    if query_intent == "complaint":
        return f"🐛 Complaints of {timeframe_label}:\n\n"
    elif query_intent == "positive":
        return f"🌻 Positive Feedback of {timeframe_label}:\n\n"
    elif query_intent == "suggestion":
        return f"💡 Suggestions & Improvement Ideas for {timeframe_label}:\n\n"
    elif query_intent == "topics":
        return f"🗣️ Top Topics of {timeframe_label}:\n\n"
    elif category_filter == PRODUCT_QUERY_CATEGORY:
        return f"💰 Product Inquiries of {timeframe_label}:\n\n"
    else:
        return f"🌾 Sentiments of {timeframe_label}:\n\n"

def build_intent_badge(query_intent, active_product, periods, active_crop=None, category_filter=None):
    """Small colored pill label for the answer — purely cosmetic metadata, UI decides how to render it."""
    subject_label = build_subject_label(active_product, active_crop)
    if periods:
        return "🔀 Comparison"
    if subject_label:
        return f"🏷️ {subject_label}"
    if query_intent == "complaint":
        return "🐛 Complaints"
    if query_intent == "positive":
        return "🌻 Positive"
    if query_intent == "suggestion":
        return "💡 Suggestions"
    if query_intent == "topics":
        return "🗣️ Top Topics"
    if category_filter == PRODUCT_QUERY_CATEGORY:
        return "💰 Product Inquiries"
    return "🌾 Sentiment Overview"

def _explicit_list_format(query_lower: str) -> bool:
    """True when the user explicitly asked for a list/bullets."""
    return bool(re.search(r'\blist(ed|ing)?\b|\bbullets?\b|\bbullet\s*points?\b', query_lower))


def _wants_products_only(query_lower: str) -> bool:
    """True when the question is specifically about products, so the model
    must be told not to present diseases or pests as if they were products."""
    return bool(re.search(r'\bproducts?\b', query_lower))


def build_system_prompt(query_intent, timeframe_label, explicit_list_format, active_product, periods, active_crop=None, output_format=None, wants_products_only=False, avoid_repeat_text=None, comparison_axis="time"):
    """ Unified prompt builder. Preserves the original prose behaviour (including the two-paragraph favorable/complaints structure for the default sentiment case) while adding: real markdown bullet formatting when the user explicitly asks to "list" something, strict single-product/crop focus, explicit period-by-period comparison instructions, and output-format overrides (table / executive summary / PPT outline). """
    product_label = active_product.title() if active_product else None
    crop_label = active_crop.title() if active_crop else None

    if active_product and active_crop:
        product_clause = (
            f"Focus EXCLUSIVELY on the product '{product_label}' as it relates to the "
            f"crop '{crop_label}'. Do NOT mention any other product or crop, even if "
            f"they appear in the data context.\n"
        )
    elif active_product:
        product_clause = (
            f"Focus EXCLUSIVELY on the product '{product_label}'. Do NOT mention, "
            f"reference, or summarize information about any other product, even if "
            f"other products appear in the data context — ignore anything not about "
            f"'{product_label}'.\n"
        )
    elif active_crop:
        product_clause = (
            f"Focus EXCLUSIVELY on the crop '{crop_label}'. Do NOT mention, reference, "
            f"or summarize information about any other crop, even if other crops appear "
            f"in the data context — ignore anything not about '{crop_label}'. Still name "
            f"every product mentioned in connection with '{crop_label}'.\n"
        )
    else:
        product_clause = (
            "Explicitly name every product mentioned in the data context along with "
            "the exact reason for the feedback.\n"
        )

    comparison_clause = ""
    if periods:
        period_names = ", ".join(p[0] for p in periods)
        first = period_names.split(', ')[0]
        if comparison_axis == "subject":
            # Comparing products/crops against each other, not time periods.
            comparison_clause = (
                f"This is a side-by-side COMPARISON of: {period_names}. "
                f"The data context below is divided into one clearly labeled section per "
                f"item. Compare them directly against each other — say which is better "
                f"received and in what respect, and call out where they differ. Refer to "
                f"each by its exact name, and cover EVERY one of them: do not answer about "
                f"only one. If the data context for one of them is empty, say plainly that "
                f"there is no feedback for it rather than omitting it silently. "
                f"End with a one-sentence bottom line, e.g. 'Overall {first} is better "
                f"regarded for <reason>.'\n"
            )
        else:
            comparison_clause = (
                f"This is a COMPARISON request across these periods: {period_names}. "
                f"The data context below is divided into clearly labeled sections, one per "
                f"period. Explicitly compare the periods against each other — call out what "
                f"increased, decreased, improved, worsened, or stayed roughly the same. "
                f"Refer to each period by its exact name. "
                f"CRITICAL: for every point you make, name the specific product it is about "
                f"(never speak only in generic sentences with no product named), and for each "
                f"period state plainly whether that product's feedback was positive/satisfactory "
                f"or negative/unsatisfactory in that period — e.g. 'In {first}, "
                f"growers were satisfied with <Product>, but in the other period they were not.' "
                f"Do this for every product that appears in the data context.\n"
            )

    intent_label = {
        "complaint":  "complaints and concerns (including root-cause issues)",
        "positive":   "positive feedback and appreciation",
        "suggestion": "grower suggestions and improvement recommendations",
        "sentiment":  "overall sentiment (both positive and negative)",
        "topics":     "the most frequently discussed topics and themes",
    }[query_intent]

    opening_hint = {
        "complaint":  f"e.g. 'Here are the complaints for {timeframe_label}:'",
        "positive":   f"e.g. 'The positive feedback for {timeframe_label} looks great!'",
        "suggestion": f"e.g. 'Here are the grower suggestions for {timeframe_label}:'",
        "sentiment":  f"e.g. 'Here is the sentiment overview for {timeframe_label}:'",
        "topics":     f"e.g. 'Here's what growers are talking about most for {timeframe_label}:'",
    }[query_intent]
    subject_label = build_subject_label(active_product, active_crop)
    if subject_label:
        opening_hint = f"e.g. 'Here's what growers are saying about {subject_label}:'"

    output_format_clause = ""
    if output_format == "table":
        output_format_clause = (
            "OUTPUT FORMAT OVERRIDE: Format your ENTIRE response as a markdown table "
            "with columns '| Category | Feedback |'. One data point per row. No prose "
            "outside the table.\n"
        )
    elif output_format == "exec_summary":
        output_format_clause = (
            "OUTPUT FORMAT OVERRIDE: Write a one-page executive summary with these "
            "bold section headers on their own lines, in order: '*Headline:*' (one "
            "sentence takeaway), '*Key Insights:*' (3-5 bullet points, one specific "
            "point each), '*Recommendation:*' (1-2 sentences on what to do next).\n"
        )
    elif output_format == "ppt":
        output_format_clause = (
            "OUTPUT FORMAT OVERRIDE: Format your response as a PowerPoint-ready slide "
            "outline. Use '*Slide 1: <short title>*' on its own line, followed by "
            "3-5 short bullet points (each starting with '- '), then a blank line "
            "before the next slide if more than one slide is warranted. Keep every "
            "bullet short enough to fit on a slide (under 15 words).\n"
        )

    if explicit_list_format:
        format_clause = (
            "Format your ENTIRE response as a real markdown bullet list. Every single "
            "bullet MUST start on its own new line with a dash and a space: '- '. "
            "Never write '•' and never put more than one bullet on the same line. "
            "Each bullet must be one specific, concrete point (one product/issue per "
            "bullet) — no paragraphs, no prose outside the list.\n"
        )
        if query_intent == "sentiment" and not periods:
            format_clause += (
                "Group the bullets under two bold headers on their own lines: "
                "'*Positive:*' followed by positive bullets, then a blank line, then "
                "'*Negative:*' followed by negative bullets.\n"
            )
        if periods:
            format_clause += (
                "Group the bullets under one bold header per period (using the exact "
                "period names given above, each on its own line), followed by that "
                "period's bullets, with a blank line between groups.\n"
            )
        structure_clause = ""
    else:
        format_clause = (
            "Respond in natural, flowing prose — no bullet points, no markdown lists, "
            "no asterisks. Sound like a helpful chatbot, not a formal report. Keep it "
            "concise.\n"
        )
        if query_intent == "sentiment" and not periods and not active_product:
            structure_clause = (
                "Structure your response in exactly two short paragraphs:\n"
                "Paragraph 1 — Favorable Sentiments: summarize positive trends.\n"
                "Paragraph 2 — Complaints & Concerns: summarize issues.\n"
                "Each paragraph should be 3-5 sentences max.\n"
            )
        elif query_intent == "topics":
            structure_clause = (
                "Structure your response as one short paragraph per topic (3-6 "
                "topics total, most-discussed first), each naming the topic and "
                "summarizing what's said about it in 1-2 sentences.\n"
            )
        else:
            structure_clause = "Keep the response to 4-6 sentences max.\n"

    topics_clause = ""
    if query_intent == "topics":
        topics_clause = (
            "This is a TOPICS/THEMES request, not a sentiment request: identify "
            "the 3-6 most frequently occurring topics or themes in the data "
            "context below (e.g. pricing, product performance, packaging, "
            "availability, application timing) and briefly describe what "
            "growers are saying about each one. Do NOT frame this as "
            "positive-vs-negative sentiment analysis — focus on WHAT is being "
            "discussed, not whether it is good or bad. Order topics from most "
            "to least frequently mentioned where you can tell.\n"
        )

    # An explicit output-format request (table / exec summary / PPT outline)
    # always wins over the default bullet/prose formatting instructions.
    if output_format_clause:
        format_clause = output_format_clause
        structure_clause = ""

    system_prompt = (
        "You are a smart, friendly chatbot analyst for Syngenta, an agriculture company. "
        "STRICT GROUNDING RULE — READ CAREFULLY: You must use ONLY the information given "
        "to you in the 'Data Context' block in the user's message. You have general "
        "knowledge about real Syngenta/agriculture products from your training — you must "
        "IGNORE all of that here. Do NOT invent, assume, guess, or add any product name, "
        "complaint, statistic, or feedback point that is not explicitly written in the Data "
        "Context, even if it sounds plausible or matches a real product you know about. If "
        "the Data Context contains only one point, your entire response must be based on "
        "that single point only — never pad the list with extra products or details to make "
        "it look longer or more complete. If the Data Context is empty or has nothing "
        "relevant, say so plainly instead of making something up. "
        f"Cover ONLY {intent_label} from the data context provided. "
        f"{product_clause}"
        + (
            "The user is asking specifically about PRODUCTS. Only name real "
            "Syngenta product/brand names (e.g. Isabion, Axial, Quantis, "
            "Cropwise). Do NOT list crop diseases, pests, weeds, or agronomic "
            "problems (e.g. Septoria, Blast, Armyworm, termites, weeds, "
            "yellowing) as if they were products — mention a disease/pest "
            "only in passing if it explains why a product was used, never as "
            "an item in a products list.\n"
            if wants_products_only else ""
        )
        + f"{comparison_clause}"
        f"{topics_clause}"
        f"{format_clause}"
        f"{structure_clause}"
        "ANSWER THE QUESTION THAT WAS ASKED. The user's exact question is at "
        "the end of the message below. Address that specific question first "
        "and directly — if they asked about price, lead with price; if they "
        "asked about packaging, lead with packaging; if they asked whether "
        "something improved, say whether it improved. Do NOT substitute a "
        "general overview of the subject for an answer to the question. If "
        "the Data Context genuinely does not address what they asked, say so "
        "in one plain sentence and then give the closest relevant information "
        "you do have, clearly labelled as such. "
        f"Start your response with a short, clear opening line ({opening_hint}), then "
        "continue. Write so a busy reader understands the key takeaway at first glance. "
        "Do not include bracketed dates, week labels, or raw metadata tags in the output. "
        "REMINDER: every product name and every point in your response must come directly "
        "from the Data Context above — never introduce a product or detail that isn't "
        "explicitly there."
    )

    if avoid_repeat_text:
        system_prompt += (
            "\n\nCONTINUATION REQUEST: The user already received the response below "
            "earlier in this conversation and is now explicitly asking for MORE or "
            "DIFFERENT points on the same subject — not a repeat. Do not restate any "
            "point from it in substantially the same words. Pull NEW points from the "
            "Data Context that weren't already covered. If nothing further is "
            "supported by the Data Context, say plainly that there isn't anything "
            "further to add rather than repeating the earlier points.\n\n"
            f"Previous response already given:\n{avoid_repeat_text}"
        )

    return system_prompt
