"""
Voice of Grower — framework-agnostic core.

Everything in this module is plain Python: no Streamlit imports, no UI calls.
It holds the constants, retrieval/ranking/trend logic, the Excel ingestion
parser, and the PPTX/Excel builders that power the chatbot. The Streamlit
app (app.py) is a thin UI wrapper around this module; a future FastAPI
backend can import the exact same module and get identical behavior.

Two-phase chat API (mirrors what any UI needs — get a prompt, stream the
LLM, then finalize with charts/downloads):

    state = process_chat_query(user_query, pinecone_api_key, groq_api_key)
    if state["kind"] in ("blocked", "no_key", "ranking", "trend", "no_data"):
        # state["reply"] is ready to display; state.get("downloads") may hold
        # ready CSV/Excel/PPTX bytes and state.get("chart") the chart data.
        ...
    elif state["kind"] == "normal":
        # Stream state["system_prompt"] / state["user_prompt"] through Groq
        # yourself (so each UI can render tokens its own way), then:
        result = finalize_normal_response(state, full_response_text)
        # result["final_reply"], result["chart"], result["downloads"]
"""

import re
import os
from io import BytesIO
from collections import Counter

import pandas as pd
from pinecone import Pinecone
from groq import Groq
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PINECONE_INDEX_NAME = "chatbot"
EMBEDDING_DIMENSION = 384
GROQ_MODEL = "llama-3.1-8b-instant"

MONTH_TYPO_FIX = {
    "Feburary": "February", "Febuary": "February",
    "Septembar": "September", "Septmber": "September",
    "Octobar": "October",    "Novembar": "November",
    "Decembar": "December",  "Januray": "January",
    "Janaury": "January",    "Marck": "March"
}

MONTH_MAP = {
    "january": "January", "february": "February", "march": "March",
    "april": "April",     "may": "May",            "june": "June",
    "july": "July",       "august": "August",      "september": "September",
    "october": "October", "november": "November",  "december": "December",
    "jan": "January",     "feb": "February",       "mar": "March",
    "apr": "April",       "jun": "June",           "jul": "July",
    "aug": "August",      "sep": "September",      "oct": "October",
    "nov": "November",    "dec": "December"
}

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12
}

CATEGORY_NORMALIZE = {
    "product queries":              "Product Queries",
    "problem/advisory":             "Problem/Advisory",
    "problem advisory":             "Problem/Advisory",
    "positive feedback":            "Positive Feedback",
    "complaint/negative feedback":  "Complaint/Negative Feedback",
    "complaint negative feedback":  "Complaint/Negative Feedback",
    "complaints":                   "Complaint/Negative Feedback",
    "negative feedback":            "Complaint/Negative Feedback",
    "suggestion":                   "Suggestions",
    "suggestions":                  "Suggestions",
    "others":                       "Others",
    "other":                        "Others"
}

POSITIVE_CATEGORIES = {"Positive Feedback"}
# Problem/Advisory rows are grower-reported issues/concerns just like
# Complaint/Negative Feedback — both feed the "complaint" / "root cause" /
# "top concerns" style queries.
NEGATIVE_CATEGORIES = {"Complaint/Negative Feedback", "Problem/Advisory"}
SUGGESTION_CATEGORY  = "Suggestions"

EMPTY_VALUES = {
    'nan', 'none', '', 'null', '-', 'n/a', 'na',
    'not filled', 'not available', 'no data', '0', 'tbd', 'pending'
}

# Known products — matched first (fast path). Extend freely.
PRODUCT_LIST = [
    "cropwise", "quantis", "isabion", "allymax", "axial", "walter", "kaho",
    "solubor", "amistar", "incipio", "simodis", "solvigo", "rifit",
    "logran", "cruiser", "enrich", "virtako", "proclaim", "thiovet",
    "pendimethalin", "polytrin", "chlorpyrifos", "glyphosate", "tilt",
    "actara", "alika", "ridomil", "score", "folicur", "miraculan",
    "dual gold", "naya potash"
]

# Known crops — matched first (fast path) for crop-wise analysis / filtering.
CROP_LIST = [
    "wheat", "rice", "paddy", "cotton", "maize", "corn", "sugarcane",
    "soybean", "soyabean", "groundnut", "mustard", "canola", "potato",
    "tomato", "onion", "chilli", "chili", "gram", "chickpea", "pea", "peas",
    "banana", "grape", "grapes", "sunflower", "okra", "lady finger",
    "ladyfinger", "barley", "jowar", "bajra", "cabbage", "cauliflower",
    "brinjal", "cucumber", "watermelon", "muskmelon", "melon", "mango",
    "citrus", "orange", "apple", "turmeric", "ginger", "garlic", "sesame",
    "castor", "tobacco", "rose", "roses", "papaya", "guava", "pomegranate",
    "carrot", "radish", "spinach", "cumin", "coriander", "fenugreek"
]

# Diseases, pests, and agronomic problems that must NEVER be reported back to
# the user as if they were products — the raw feedback text often names them
# in the same sentence as real products (e.g. "Septoria control with Score"),
# and the capitalized-phrase heuristic in extract_product_mentions would
# otherwise tag them as brand names.
DISEASE_PEST_TERMS = [
    "septoria", "blast", "blight", "rust", "wilt", "rot", "mildew",
    "powdery mildew", "downy mildew", "virus", "mosaic", "mosaic virus",
    "aphid", "aphids", "borer", "borers", "caterpillar", "caterpillars",
    "armyworm", "armyworms", "fall armyworm", "faw", "termite", "termites",
    "rodent", "rodents", "whitefly", "whiteflies", "thrips", "mite", "mites",
    "nematode", "nematodes", "weed", "weeds", "fungus", "fungal", "bacterial",
    "bacteria", "larvae", "larva", "infestation", "infestations", "scab",
    "canker", "leaf spot", "leaf curl", "yellowing", "stunting", "wilting",
    "disease", "diseases", "pest", "pests"
]
# Flattened to individual words so multi-word phrases (e.g. "Leaf Curl Virus")
# are still caught even though only part of the phrase matches a listed term.
DISEASE_PEST_WORDS = {w for term in DISEASE_PEST_TERMS for w in term.split()}

# Extra business-domain vocabulary the strict topic guardrail must recognize
# (executive/sales/marketing/digital/advanced-analytics use cases, plus the
# various output-format requests such as tables, charts, and exports).
BUSINESS_KEYWORDS = [
    "suggestion", "suggestions", "recommend", "recommendation",
    "recommendations", "improvement", "improvements", "expectation",
    "expectations", "root", "cause", "causes", "trend", "trends",
    "trending", "rank", "ranking", "ranked", "top", "table", "excel",
    "export", "download", "ppt", "powerpoint", "presentation", "slide",
    "slides", "chart", "graph", "visualize", "visualise", "visualization",
    "visualisation", "dashboard", "executive", "summary", "summarize",
    "sales", "marketing", "digital", "campaign", "campaigns",
    "misconception", "misconceptions", "awareness", "yoy", "quarter",
    "quarterly", "insight", "insights", "strategic", "priority",
    "priorities", "forecast", "predict", "prediction", "region", "regions",
    "crop", "crops", "emerging", "satisfaction", "frequency", "frequent",
    "common", "pattern", "patterns", "hidden", "platform", "platforms",
    "online", "support", "experience", "customer", "recommended", "most",
    "highest", "significant", "significantly", "increased", "improve",
    "business", "monthly", "yearly", "annual", "annually", "breakdown",
    "revenue", "growth", "performance", "analytics", "kpi", "kpis"
]

# Generic words that should never be mistaken for a product name during
# dynamic (fallback) product detection.
PRODUCT_STOPWORDS = {
    "sentiment", "sentiments", "feedback", "feedbacks", "product", "products",
    "syngenta", "app", "price", "unavailability", "complaint", "complaints",
    "positive", "negative", "overview", "overall", "general", "summary",
    "both", "analysis", "grower", "growers", "advisory", "week", "weeks",
    "list", "listed", "listing", "bullet", "bullets", "compare", "comparison",
    "versus", "give", "show", "tell", "what", "are", "about", "the", "for",
    "and", "of", "in", "on", "me", "please", "this", "that", "month",
    "months", "year", "years", "data", "record", "records", "issue",
    "issues", "problem", "problems", "concern", "concerns", "appreciation",
    "praise", "favorable", "satisfied", "first", "second", "third",
    "fourth", "fifth", "point", "points", "wise", "chatbot", "yield",
    # common connector / filler words that must never be treated as a
    # product name during dynamic detection
    "out", "down", "up", "into", "onto", "with", "from", "than", "then",
    "just", "only", "also", "very", "much", "many", "more", "most", "some",
    "such", "need", "needs", "want", "wants", "know", "get", "gets", "got",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "not", "no", "yes", "okay", "ok", "thanks", "thank", "you", "your",
    "our", "their", "his", "her", "its", "all", "any", "each", "every",
    "who", "whom", "which", "when", "where", "why", "how", "does", "did",
    "has", "have", "had", "was", "were", "been", "being", "here", "there",
    "these", "those", "over", "under", "again", "still", "yet", "now",
    "provide", "write", "respond", "answer", "reply", "query", "ask",
    "asking", "kindly", "regarding", "specific", "particular", "quick",
    "quickly", "brief", "detail", "details", "info", "information", "one",
    "two", "three", "four", "five", "recent", "latest", "last", "current",
    "previous", "next", "prior", "season", "seasons", "compared", "across",
    "during", "within", "improved"
} | set(MONTH_MAP.keys()) | set(BUSINESS_KEYWORDS) | set(DISEASE_PEST_TERMS)

ALLOWED_GUARDRAIL_KEYWORDS = set([
    "sentiment", "sentiments", "feedback", "feedbacks", "product", "products",
    "syngenta", "cropwise", "app", "price", "unavailability", "january",
    "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "oct", "nov", "dec", "2023", "2024", "2025", "2026", "2027",
    "complaint", "complaints", "positive", "negative", "grower", "growers",
    "advisory", "week", "weeks", "1st", "2nd", "3rd", "4th", "5th", "first",
    "second", "third", "fourth", "fifth", "issues", "concerns", "problems",
    "appreciation", "praise", "compare", "comparison", "versus", "list"
]) | set(PRODUCT_LIST) | set(CROP_LIST) | set(BUSINESS_KEYWORDS)

SUGGESTED_PROMPTS = {
    "📊 Executive": [
        "What are the top 10 grower concerns reported during the last 30 days?",
        "Summarize the major grower insights for the last quarter in one page.",
        "What are the emerging trends compared to the previous year?",
    ],
    "🌾 Crop-wise": [
        "What are the top issues reported in rice during 2026?",
        "Compare grower feedback for wheat between 2025 and 2026.",
        "Show the sentiment analysis for cotton growers.",
    ],
    "🏷️ Product-wise": [
        "Which Syngenta products received the highest positive feedback?",
        "List the products with the highest complaint frequency.",
        "Compare customer sentiment for Tilt and Isabion.",
    ],
    "🐛 Complaints": [
        "What are the top five complaints received in the last six months?",
        "Identify the root causes of the most common complaints.",
        "Show the monthly complaint trend for the past three years.",
    ],
    "🌻 Sentiment": [
        "Show overall grower sentiment by month.",
        "Compare positive, neutral, and negative sentiment across crops.",
        "Display sentiment trends in chart and table format.",
    ],
    "💡 Suggestions": [
        "What suggestions have growers provided for rice cultivation?",
        "What are the most common product improvement recommendations?",
        "Summarize grower expectations from Syngenta.",
    ],
    "💰 Sales": [
        "Which products receive the highest number of price inquiries?",
        "Which crops generate the highest sales-related questions?",
    ],
    "📣 Marketing & Digital": [
        "What misconceptions do growers commonly have about our products?",
        "What are the most common complaints regarding digital platforms or systems?",
    ],
    "🔮 Advanced & Rankings": [
        "Which crop generated the highest number of complaints?",
        "Generate the Top 10 business insights from the last three years.",
        "Based on all historical data, recommend the top five strategic actions Syngenta should prioritize for the next season.",
    ],
    "📄 Output formats": [
        "Show grower suggestions for rice in a table.",
        "Generate an executive summary of complaints for 2026.",
        "Create a PowerPoint-ready summary of positive feedback for Isabion.",
    ],
}

# ==========================================
# UTILITIES
# ==========================================

def find_category_column(df_columns):
    for col in df_columns:
        col_clean = re.sub(r'\s+', '', str(col)).lower()
        if 'categ' in col_clean:
            return col
    return None


def infer_year_for_sheet(sheet_name: str, all_sheet_names: list) -> str | None:
    """ Sheets that carry an explicit 4-digit year in their name are trusted directly. Older legacy sheets (e.g. "Jan till June", "July till December") have no year in the name at all — for those, infer the year from neighboring dated sheets: a sheet appearing BEFORE the first explicitly-dated sheet is assumed to be from the year immediately preceding that dated sheet (legacy history predates the dated sheets); one appearing AFTER the last dated sheet is assumed to follow it by one year. """
    direct = re.search(r'(20\d{2})', sheet_name.strip())
    if direct:
        return direct.group(0)

    dated = [
        (i, int(m.group(0)))
        for i, name in enumerate(all_sheet_names)
        for m in [re.search(r'(20\d{2})', name.strip())] if m
    ]
    if not dated:
        return None

    try:
        my_index = all_sheet_names.index(sheet_name) if sheet_name in all_sheet_names else None
        if my_index is None:
            for i, name in enumerate(all_sheet_names):
                if name.strip() == sheet_name:
                    my_index = i
                    break
    except ValueError:
        my_index = None

    if my_index is None:
        return None

    later = [year for i, year in dated if i > my_index]
    if later:
        return str(min(later) - 1)

    earlier = [year for i, year in dated if i < my_index]
    if earlier:
        return str(max(earlier) + 1)

    return None


def normalize_category(raw_val):
    if not raw_val:
        return None
    cleaned = re.sub(r'\s+', ' ', str(raw_val)).strip().lower()
    if cleaned in CATEGORY_NORMALIZE:
        return CATEGORY_NORMALIZE[cleaned]
    return str(raw_val).strip()


def is_empty_cell(value: str) -> bool:
    return value.strip().lower() in EMPTY_VALUES or value.strip() == ""


def split_bullets(cell_text: str) -> list[str]:
    lines = cell_text.split('\n')
    bullets = []
    for line in lines:
        clean = re.sub(r'^[\s•●·\-–—]+', '', line).strip()
        if clean and clean.lower() not in EMPTY_VALUES and len(clean) > 3:
            bullets.append(clean)
    return bullets


def extract_month_from_col(col: str) -> str:
    match = re.search(
        r'(january|february|march|april|may|june|july|august'
        r'|september|october|november|december'
        r'|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)',
        col.lower()
    )
    if not match:
        return "Unknown"
    raw  = match.group(0).capitalize()
    full = MONTH_MAP.get(raw.lower(), raw)
    return MONTH_TYPO_FIX.get(full, full)


def get_latest_year_from_index(index) -> str:
    try:
        dummy_vector = [0.0] * EMBEDDING_DIMENSION
        results = index.query(vector=dummy_vector, top_k=10, include_metadata=True)
        years = []
        for match in results.get("matches", []):
            year = match.get("metadata", {}).get("year")
            if year and year.isdigit():
                years.append(int(year))
        if years:
            return str(max(years))
    except Exception:
        pass
    return "2026"


def get_max_week_label(index, month, year) -> str | None:
    """Find the actual latest week label (e.g. '5th Week') stored in the dataset for the given month/year, so 'last week' / 'latest week' queries resolve to a real week instead of a fixed guess."""
    filter_conditions = {}
    if month:
        filter_conditions["month"] = {"$eq": month}
    if year:
        filter_conditions["year"] = {"$eq": year}
    try:
        dummy_vector = [0.0] * EMBEDDING_DIMENSION
        results = index.query(
            vector=dummy_vector, top_k=200, include_metadata=True,
            filter=filter_conditions if filter_conditions else None
        )
        weeks = [m.get("metadata", {}).get("week", "") for m in results.get("matches", [])]
        weeks = [w for w in weeks if w]
        if not weeks:
            return None

        def week_num(w):
            match = re.search(r'(\d+)', w)
            return int(match.group(1)) if match else -1

        return max(set(weeks), key=week_num)
    except Exception:
        return None


def query_pinecone_for_timeframe(index, query_vector, month, year, week, query_intent="sentiment", top_k=100, category_filter=None):
    filter_conditions = {}
    if month:
        filter_conditions["month"] = {"$eq": month}
    if year:
        filter_conditions["year"]  = {"$eq": year}

    # ── CATEGORY FILTER (e.g. "Suggestions") TAKES PRIORITY OVER SENTIMENT ──
    if category_filter:
        filter_conditions["category"] = {"$eq": category_filter}
    # ── SENTIMENT FILTER AT DATABASE LEVEL ──
    elif query_intent == "positive":
        filter_conditions["sentiment"] = {"$eq": "positive"}
    elif query_intent == "complaint":
        filter_conditions["sentiment"] = {"$eq": "negative"}

    metadata_filter = filter_conditions if filter_conditions else None

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter
    )
    matches = results.get("matches", [])

    ORDINAL_MAP = {
        "first": "1st", "second": "2nd",
        "third": "3rd", "fourth": "4th", "fifth": "5th"
    }

    positive_bullets = []
    negative_bullets = []
    neutral_bullets  = []

    # Track exact text already added (case-insensitive) so the same feedback
    # point is never sent to the model twice — this covers duplicate vectors
    # from re-ingesting the same sheet, or the same line matching more than
    # one week column.
    seen_positive = set()
    seen_negative = set()
    seen_neutral  = set()

    for m in matches:
        md        = m.get("metadata", {})
        sentiment = md.get("sentiment", "neutral")
        value     = md.get("value", "").strip()
        w         = md.get("week", "")
        category  = md.get("category", "")

        if not value or value.lower() in EMPTY_VALUES:
            continue

        if week:
            dw = week.lower()
            for word, num in ORDINAL_MAP.items():
                dw = dw.replace(word, num)
            if dw not in w.lower():
                continue

        entry = f"{category}: {value}"
        dedupe_key = entry.strip().lower()

        if sentiment == "positive":
            if dedupe_key not in seen_positive:
                seen_positive.add(dedupe_key)
                positive_bullets.append(entry)
        elif sentiment == "negative":
            if dedupe_key not in seen_negative:
                seen_negative.add(dedupe_key)
                negative_bullets.append(entry)
        else:
            if dedupe_key not in seen_neutral:
                seen_neutral.add(dedupe_key)
                neutral_bullets.append(entry)

    return positive_bullets, negative_bullets, neutral_bullets


def extract_all_months(query_lower: str) -> list[str]:
    """Return every distinct month mentioned in the query, in the order first seen."""
    found = []
    for shortcut in sorted(MONTH_MAP.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(shortcut) + r'\b', query_lower):
            full = MONTH_MAP[shortcut]
            if full not in found:
                found.append(full)
    return found


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


def detect_product_known(query_lower: str) -> str | None:
    """Fast path: match against the curated PRODUCT_LIST."""
    for product in PRODUCT_LIST:
        if re.search(r'\b' + re.escape(product) + r'\b', query_lower):
            return product
    return None


def detect_product_dynamic(query_lower: str, index, pc) -> str | None:
    """ Fallback path for products NOT in PRODUCT_LIST. Pulls candidate word(s) out of the query (skipping common/sentiment/month/filler words), then checks a targeted Pinecone probe — embedding the candidate itself, not the user's raw question — to see if it genuinely appears inside the ingested feedback text. Using a dedicated embedding per candidate (rather than reusing the original query's embedding) means detection no longer depends on how the question happens to be phrased: "tell me about X" and "give me feedback of X" now behave identically. Multi-word product names (e.g. "Naya Potash") are tried as a full phrase first, then as individual words as a fallback. """
    candidates = [
        w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        if w not in PRODUCT_STOPWORDS
    ]
    if not candidates:
        return None

    # Try the full multi-word phrase first (handles "Naya Potash"-style names),
    # then fall back to individual candidate words.
    ordered_candidates = []
    if len(candidates) >= 2:
        ordered_candidates.append(" ".join(candidates))
    ordered_candidates.extend(candidates)

    for cand in ordered_candidates:
        try:
            probe_embed = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[f"{cand} product feedback sentiment"],
                parameters={"input_type": "query", "dimension": EMBEDDING_DIMENSION}
            )
            probe_vector = probe_embed[0].values
            probe = index.query(vector=probe_vector, top_k=50, include_metadata=True)
            blob = " ".join(
                str(m.get("metadata", {}).get("value", "")) for m in probe.get("matches", [])
            ).lower()
        except Exception:
            continue

        if cand.lower() in blob:
            return cand
    return None


def filter_bullets_by_product(bullets: list[str], product: str) -> list[str]:
    """Keep only bullets that actually reference the requested product."""
    return [b for b in bullets if product.lower() in b.lower()]


def detect_crop(query_lower: str) -> str | None:
    """Fast path: match against the curated CROP_LIST (closed vocabulary, no dynamic probe needed)."""
    for crop in CROP_LIST:
        if re.search(r'\b' + re.escape(crop) + r'\b', query_lower):
            return crop
    return None


def filter_bullets_by_crop(bullets: list[str], crop: str) -> list[str]:
    """Keep only bullets that actually reference the requested crop."""
    return [b for b in bullets if crop.lower() in b.lower()]


def extract_crops(text: str) -> list[str]:
    """Tag every known crop mentioned in a feedback bullet, for ingestion-time metadata."""
    text_lower = text.lower()
    found = []
    for crop in CROP_LIST:
        if re.search(r'\b' + re.escape(crop) + r'\b', text_lower):
            label = crop.title()
            if label not in found:
                found.append(label)
    return found


def extract_product_mentions(text: str) -> list[str]:
    """ Tag likely product/brand names mentioned in a feedback bullet, for ingestion-time metadata used by deterministic ranking ("which product received the highest complaints"). Heuristic: capitalized word sequences (1-3 words) that aren't generic English/stopwords/month names — this generalizes beyond the curated PRODUCT_LIST to any brand name that shows up in the data without needing to hardcode every product. """
    candidates = re.findall(r'\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2})\b', text)
    out, seen = [], set()
    for cand in candidates:
        words = cand.split()
        if all(w.lower() in PRODUCT_STOPWORDS or w.lower() in MONTH_MAP for w in words):
            continue
        if any(w.lower() in DISEASE_PEST_WORDS for w in words):
            continue
        if cand.strip().lower() in ("syngenta",):
            continue
        key = cand.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(cand.strip())
    return out


def build_comparison_periods(all_months, all_years, all_weeks, index):
    """ Build a list of (label, month, year, week) tuples describing each period to compare. The dimension with 2+ distinct values becomes the axis of comparison; other dimensions are held fixed. No explicit "compare" keyword is required — mentioning two+ months/years/weeks is enough. """
    periods = []

    if len(all_years) >= 2:
        month = all_months[0] if all_months else None
        week  = all_weeks[0] if all_weeks else None
        for y in all_years:
            label = f"{(month + ' ') if month else ''}{y}"
            periods.append((label, month, y, week))

    elif len(all_months) >= 2:
        year = all_years[0] if all_years else get_latest_year_from_index(index)
        week = all_weeks[0] if all_weeks else None
        for m in all_months:
            label = f"{m} {year}"
            periods.append((label, m, year, week))

    elif len(all_weeks) >= 2:
        month = all_months[0] if all_months else None
        year  = all_years[0] if all_years else get_latest_year_from_index(index)
        for w in all_weeks:
            label = f"{w} week" + (f" of {month}" if month else "") + f" {year}"
            periods.append((label, month, year, w))

    return periods


def detect_aggregation_request(query_lower: str) -> str | None:
    """ Detects queries that want a deterministic, counted ranking ("which crop generated the highest number of complaints", "products with the highest complaint frequency") rather than free-form LLM summarization. Returns 'crop' or 'product' — the metadata field to rank by — or None. Kept separate from the normal retrieval flow because counting must be exact (computed in Python from real metadata tags), never left to the LLM to eyeball from a handful of retrieved bullets. """
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
    if 'crop' in query_lower:
        return 'crop'
    if 'product' in query_lower:
        return 'product'
    return None


def fetch_matches_for_aggregation(index, filter_conditions, top_k=10000):
    """ Broad, non-semantic fetch (dummy vector) used purely for exact counting/ranking over metadata tags. top_k is set high (Pinecone's practical ceiling) because a zero vector carries no similarity signal — Pinecone returns matches in whatever internal order it likes, so a small top_k risks silently missing the tagged subset (e.g. only a minority of complaint records mention a specific crop by name) rather than giving a true representative sample. """
    dummy_vector = [0.0] * EMBEDDING_DIMENSION
    results = index.query(
        vector=dummy_vector, top_k=top_k, include_metadata=True,
        filter=filter_conditions if filter_conditions else None
    )
    return results.get("matches", [])


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


def detect_trend_request(query_lower: str) -> bool:
    """ Detects requests for a monthly trend / growth-over-time breakdown (as opposed to a single-period or comparison-of-two-periods question). Deliberately conservative — bare words like "sales" or "products" alone are far too common in ordinary questions to trigger a full trend workup, so this requires an explicit temporal/trend framing. """
    trend_phrases = [
        'trend', 'trends', 'trending', 'month over month', 'month-over-month',
        'monthly trend', 'over time', 'growth', 'analytics over time',
        'performance over', 'monthly totals', 'monthly breakdown'
    ]
    return any(p in query_lower for p in trend_phrases)


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


def build_pptx_report(title, subtitle, exec_summary_lines, kpis, chart_title, chart_labels, chart_values,
                       chart_type="column", table_headers=None, table_rows=None,
                       insights=None, recommendations=None):
    """ Build a professional PPTX deck: Title -> Executive Summary -> Key KPIs -> Chart -> Table -> Key Insights & Recommendations. Uses PowerPoint's native chart/table shapes (not an embedded image) so the deck stays fully editable in PowerPoint. Returns raw bytes. """
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    title_layout = prs.slide_layouts[0]
    bullet_layout = prs.slide_layouts[1]
    blank_layout = prs.slide_layouts[6]

    # ── Slide 1: Title ──
    slide = prs.slides.add_slide(title_layout)
    slide.shapes.title.text = title
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle

    # ── Slide 2: Executive Summary ──
    slide = prs.slides.add_slide(bullet_layout)
    slide.shapes.title.text = "Executive Summary"
    tf = slide.placeholders[1].text_frame
    tf.clear()
    lines = exec_summary_lines or ["No summary available."]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(18)

    # ── Slide 3: Key KPIs ──
    slide = prs.slides.add_slide(bullet_layout)
    slide.shapes.title.text = "Key KPIs"
    tf = slide.placeholders[1].text_frame
    tf.clear()
    kpi_items = list((kpis or {}).items()) or [("No KPIs available", "")]
    for i, (k, v) in enumerate(kpi_items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(20)
        p.font.bold = True

    # ── Slide 4: Chart ──
    if chart_labels and chart_values:
        slide = prs.slides.add_slide(blank_layout)
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12), Inches(0.6))
        title_box.text_frame.text = chart_title
        title_box.text_frame.paragraphs[0].font.size = Pt(24)
        title_box.text_frame.paragraphs[0].font.bold = True

        chart_data = CategoryChartData()
        chart_data.categories = chart_labels
        chart_data.add_series(chart_title, chart_values)
        xl_type = XL_CHART_TYPE.LINE_MARKERS if chart_type == "line" else XL_CHART_TYPE.COLUMN_CLUSTERED
        slide.shapes.add_chart(xl_type, Inches(0.75), Inches(1.0), Inches(11.8), Inches(6.0), chart_data)

    # ── Slide 5: Table ──
    if table_headers and table_rows:
        slide = prs.slides.add_slide(blank_layout)
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12), Inches(0.6))
        title_box.text_frame.text = "Supporting Data"
        title_box.text_frame.paragraphs[0].font.size = Pt(24)
        title_box.text_frame.paragraphs[0].font.bold = True

        MAX_ROWS = 15
        rows_to_show = table_rows[:MAX_ROWS]
        n_rows = len(rows_to_show) + 1
        n_cols = len(table_headers)
        table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(0.5), Inches(1.0), Inches(12.3), Inches(6.0))
        table = table_shape.table
        for c, header in enumerate(table_headers):
            table.cell(0, c).text = str(header)
        for r, row in enumerate(rows_to_show, start=1):
            for c, val in enumerate(row):
                cell_text = str(val)
                table.cell(r, c).text = cell_text[:300]

    # ── Slide 6: Key Insights & Recommendations ──
    if insights or recommendations:
        slide = prs.slides.add_slide(bullet_layout)
        slide.shapes.title.text = "Key Insights & Recommendations"
        tf = slide.placeholders[1].text_frame
        tf.clear()
        first = True
        for point in (insights or []):
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = f"Insight: {point}"
            p.font.size = Pt(16)
        for point in (recommendations or []):
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = f"Recommendation: {point}"
            p.font.size = Pt(16)
            p.font.bold = True

    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()


def build_excel_bytes(rows: list[dict]) -> bytes:
    """Turn a list of flat dict rows into an .xlsx file's raw bytes."""
    buf = BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def build_csv_bytes(rows: list[dict], columns: list[str] | None = None) -> bytes:
    df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def split_into_points(text: str, max_points: int = 6) -> list[str]:
    """Break an LLM prose/bullet response into short standalone points for slide bullets."""
    lines = [l.strip(" -*").strip() for l in text.split("\n") if l.strip(" -*").strip()]
    if len(lines) >= 2:
        return lines[:max_points]
    # Fall back to sentence-splitting for single-paragraph prose responses.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()][:max_points]


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


def build_header(query_intent, timeframe_label, active_product, periods, active_crop=None):
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
        suffix = f" ({timeframe_label})" if timeframe_label != "the requested period" else ""
        if query_intent == "complaint":
            return f"🐛 Complaints about {subject_label}{suffix}:\n\n"
        elif query_intent == "positive":
            return f"🌻 Positive Feedback about {subject_label}{suffix}:\n\n"
        elif query_intent == "suggestion":
            return f"💡 Suggestions about {subject_label}{suffix}:\n\n"
        else:
            return f"🌾 {subject_label} — Sentiment Overview{suffix}:\n\n"

    if query_intent == "complaint":
        return f"🐛 Complaints of {timeframe_label}:\n\n"
    elif query_intent == "positive":
        return f"🌻 Positive Feedback of {timeframe_label}:\n\n"
    elif query_intent == "suggestion":
        return f"💡 Suggestions & Improvement Ideas for {timeframe_label}:\n\n"
    else:
        return f"🌾 Sentiments of {timeframe_label}:\n\n"


def build_intent_badge(query_intent, active_product, periods, active_crop=None):
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
    return "🌾 Sentiment Overview"


def build_system_prompt(query_intent, timeframe_label, explicit_list_format, active_product, periods, active_crop=None, output_format=None, wants_products_only=False):
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
        comparison_clause = (
            f"This is a COMPARISON request across these periods: {period_names}. "
            f"The data context below is divided into clearly labeled sections, one per "
            f"period. Explicitly compare the periods against each other — call out what "
            f"increased, decreased, improved, worsened, or stayed roughly the same. "
            f"Refer to each period by its exact name. "
            f"CRITICAL: for every point you make, name the specific product it is about "
            f"(never speak only in generic sentences with no product named), and for each "
            f"period state plainly whether that product's feedback was positive/satisfactory "
            f"or negative/unsatisfactory in that period — e.g. 'In {period_names.split(', ')[0]}, "
            f"growers were satisfied with <Product>, but in the other period they were not.' "
            f"Do this for every product that appears in the data context.\n"
        )

    intent_label = {
        "complaint":  "complaints and concerns (including root-cause issues)",
        "positive":   "positive feedback and appreciation",
        "suggestion": "grower suggestions and improvement recommendations",
        "sentiment":  "overall sentiment (both positive and negative)"
    }[query_intent]

    opening_hint = {
        "complaint":  f"e.g. 'Here are the complaints for {timeframe_label}:'",
        "positive":   f"e.g. 'The positive feedback for {timeframe_label} looks great!'",
        "suggestion": f"e.g. 'Here are the grower suggestions for {timeframe_label}:'",
        "sentiment":  f"e.g. 'Here is the sentiment overview for {timeframe_label}:'"
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
        else:
            structure_clause = "Keep the response to 4-6 sentences max.\n"

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
        f"{format_clause}"
        f"{structure_clause}"
        f"Start your response with a short, clear opening line ({opening_hint}), then "
        "continue. Write so a busy reader understands the key takeaway at first glance. "
        "Do not include bracketed dates, week labels, or raw metadata tags in the output. "
        "REMINDER: every product name and every point in your response must come directly "
        "from the Data Context above — never introduce a product or detail that isn't "
        "explicitly there."
    )
    return system_prompt


# ==========================================
# EXCEL INGESTION (no UI calls — raises on error, returns a summary dict)
# ==========================================

def _make_metadata_payload(inferred_year, row_month, week_label, category, bullet):
    is_positive = category in POSITIVE_CATEGORIES
    is_negative = category in NEGATIVE_CATEGORIES
    context_chunk = (
        f"Year: {inferred_year}. "
        f"Month: {row_month}. "
        f"Week: {week_label}. "
        f"Case Category: {category}. "
        f"Feedback: {bullet}."
    )
    return context_chunk, {
        "text":      context_chunk,
        "month":     row_month,
        "year":      inferred_year,
        "week":      week_label,
        "category":  category,
        "sentiment": (
            "positive" if is_positive
            else "negative" if is_negative
            else "neutral"
        ),
        "value":    bullet,
        "crop":     ",".join(extract_crops(bullet)),
        "products": ",".join(extract_product_mentions(bullet)),
    }


def run_ingestion(file_bytes: bytes, pinecone_api_key: str) -> dict:
    """ Parse the Voice of Grower workbook (both known sheet layouts), tag each feedback bullet with crop/product/sentiment/category metadata, embed it, and upsert into Pinecone. Returns {"total_records": int}. Raises ValueError if nothing was found, or whatever exception the Pinecone/pandas calls raise on failure — callers (Streamlit, FastAPI, a CLI) decide how to surface that. """
    excel_file = pd.ExcelFile(BytesIO(file_bytes))
    all_sheets = excel_file.sheet_names

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)

    payload_batch = []
    text_inputs_for_embedding = []
    discovered_data_summary = {}

    for sheet_name in all_sheets:
        sheet_clean = sheet_name.strip()
        inferred_year = infer_year_for_sheet(sheet_clean, all_sheets)
        if not inferred_year:
            continue

        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

        cat_col = find_category_column(df.columns)

        if cat_col:
            # ── LAYOUT A: categories are ROWS, weeks are COLUMNS ──
            week_cols = [c for c in df.columns if 'week' in c.lower()]

            for idx, row in df.iterrows():
                raw_category = row.get(cat_col, None)
                category = normalize_category(raw_category)

                if not category or str(raw_category).strip().lower() in EMPTY_VALUES:
                    continue

                for col in week_cols:
                    cell_raw = str(row[col]).strip()
                    if is_empty_cell(cell_raw):
                        continue

                    bullets = split_bullets(cell_raw)
                    if not bullets:
                        continue

                    row_month = extract_month_from_col(col)
                    stat_key = f"{row_month} {inferred_year}"
                    discovered_data_summary[stat_key] = (
                        discovered_data_summary.get(stat_key, 0) + len(bullets)
                    )

                    for b_idx, bullet in enumerate(bullets):
                        context_chunk, metadata_payload = _make_metadata_payload(
                            inferred_year, row_month, col, category, bullet
                        )
                        clean_cat = re.sub(r'[^a-zA-Z0-9]', '', category.replace(' ', '_'))
                        clean_col = re.sub(r'[^a-zA-Z0-9]', '', col.replace(' ', '_'))
                        clean_sheet = re.sub(r'[^a-zA-Z0-9]', '', sheet_clean.replace(' ', '_'))
                        vector_id = f"v_{clean_sheet}{clean_cat}{clean_col}{idx}{b_idx}"

                        payload_batch.append({"id": vector_id, "metadata": metadata_payload})
                        text_inputs_for_embedding.append(context_chunk)

        else:
            # ── LAYOUT B: Month/Week are ROW values, categories are COLUMN headers ──
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

            header_row_idx = None
            for i in range(min(10, len(df_raw))):
                row_vals = [str(v).strip().lower() for v in df_raw.iloc[i].tolist()]
                if 'month' in row_vals:
                    header_row_idx = i
                    break
            if header_row_idx is None:
                continue

            col_map = {}
            for j, v in enumerate(df_raw.iloc[header_row_idx].tolist()):
                text = re.sub(r'\s+', ' ', str(v)).strip()
                if text and text.lower() != 'nan':
                    col_map[j] = text

            month_col_idx = next((i for i, v in col_map.items() if v.strip().lower() == 'month'), None)
            week_col_idx = next((i for i, v in col_map.items() if v.strip().lower() == 'week'), None)
            category_cols = {i: v for i, v in col_map.items() if i not in (month_col_idx, week_col_idx)}
            if month_col_idx is None or not category_cols:
                continue

            current_month = None
            current_week = None
            for r in range(header_row_idx + 1, len(df_raw)):
                row = df_raw.iloc[r]

                mval = row[month_col_idx]
                if pd.notna(mval) and str(mval).strip():
                    current_month = extract_month_from_col(str(mval).strip())

                if week_col_idx is not None:
                    wval = row[week_col_idx]
                    if pd.notna(wval) and str(wval).strip():
                        current_week = str(wval).strip()

                if not current_month:
                    continue

                week_label = f"{current_week} Week {current_month}" if current_week else current_month

                for col_idx, raw_category_name in category_cols.items():
                    category = normalize_category(raw_category_name)
                    if not category:
                        continue

                    cell_val = row[col_idx]
                    if pd.isna(cell_val):
                        continue
                    cell_raw = str(cell_val).strip()
                    if is_empty_cell(cell_raw):
                        continue

                    bullets = split_bullets(cell_raw)
                    if not bullets:
                        continue

                    stat_key = f"{current_month} {inferred_year}"
                    discovered_data_summary[stat_key] = (
                        discovered_data_summary.get(stat_key, 0) + len(bullets)
                    )

                    for b_idx, bullet in enumerate(bullets):
                        context_chunk, metadata_payload = _make_metadata_payload(
                            inferred_year, current_month, week_label, category, bullet
                        )
                        clean_cat = re.sub(r'[^a-zA-Z0-9]', '', category.replace(' ', '_'))
                        clean_week = re.sub(r'[^a-zA-Z0-9]', '', week_label.replace(' ', '_'))
                        clean_sheet = re.sub(r'[^a-zA-Z0-9]', '', sheet_clean.replace(' ', '_'))
                        vector_id = f"v_{clean_sheet}{clean_cat}{clean_week}{r}{b_idx}"

                        payload_batch.append({"id": vector_id, "metadata": metadata_payload})
                        text_inputs_for_embedding.append(context_chunk)

    total_records = len(payload_batch)
    if total_records == 0:
        raise ValueError("No records found in the uploaded workbook.")

    BATCH_LIMIT = 96
    all_vectors = []

    for i in range(0, total_records, BATCH_LIMIT):
        text_batch = text_inputs_for_embedding[i: i + BATCH_LIMIT]
        embeddings_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=text_batch,
            parameters={"input_type": "passage", "dimension": EMBEDDING_DIMENSION}
        )
        all_vectors.extend([item.values for item in embeddings_response])

    upsert_buffer = []
    for i, item in enumerate(payload_batch):
        upsert_buffer.append({
            "id": item["id"],
            "values": all_vectors[i],
            "metadata": item["metadata"]
        })
        if len(upsert_buffer) >= 50:
            index.upsert(vectors=upsert_buffer)
            upsert_buffer = []

    if upsert_buffer:
        index.upsert(vectors=upsert_buffer)

    return {"total_records": total_records, "summary": discovered_data_summary}


# ==========================================
# CHAT ORCHESTRATION (two-phase: process → stream LLM yourself → finalize)
# ==========================================

def is_query_in_scope(user_query: str) -> bool:
    """Strict topic guardrail: at least one recognized domain word must appear."""
    query_words = re.findall(r'\b\w+\b', user_query.lower())
    return any(word in ALLOWED_GUARDRAIL_KEYWORDS for word in query_words)


def process_chat_query(user_query: str, pinecone_api_key: str, groq_api_key: str | None = None) -> dict:
    """ Runs everything up to (but not including) the Groq call. Returns a dict with "kind": - "blocked": off-topic query. "reply" is ready to show. - "no_key": Pinecone not configured. "reply" is ready to show. - "ranking" / "trend" / "no_data": fully resolved without needing an LLM — "reply" (markdown), optionally "chart" ({"type","title","labels","values"}) and "downloads" ({"csv","excel","pptx"} bytes). - "normal": needs an LLM call. Contains "system_prompt" / "user_prompt" ready to send to Groq, plus everything finalize_normal_response() will need afterwards. """
    if not is_query_in_scope(user_query):
        return {
            "kind": "blocked",
            "reply": (
                "I cannot generate this response. "
                "I am strictly locked to analyzed dataset metrics "
                "and cannot find relevant information for this query."
            ),
        }

    if not pinecone_api_key:
        return {"kind": "no_key", "reply": "🤖 Execution Halted: Pinecone API key is not configured."}

    query_lower = user_query.lower()

    all_months = extract_all_months(query_lower)
    all_years = extract_all_years(query_lower)
    all_weeks = extract_all_weeks(query_lower)

    detected_month = all_months[0] if all_months else None
    detected_year = all_years[0] if all_years else None
    detected_week = all_weeks[0] if all_weeks else None

    wants_last_week = bool(re.search(r'\b(last|latest|most recent|recent)\s+week\b', query_lower))
    explicit_list_format = bool(re.search(r'\blist(ed|ing)?\b|\bbullets?\b|\bbullet\s*points?\b', query_lower))
    output_format = detect_output_format(query_lower)
    wants_products_only = bool(re.search(r'\bproducts?\b', query_lower))
    wants_trend = detect_trend_request(query_lower)
    aggregation_dimension = detect_aggregation_request(query_lower)

    complaint_keywords = [
        "complaint", "complaints", "negative feedback",
        "negative", "issues", "problems", "concerns",
        "issue", "problem", "root cause", "root causes"
    ]
    positive_keywords = [
        "positive feedback", "appreciation", "praise",
        "favorable", "satisfied"
    ]
    suggestion_keywords = [
        "suggestion", "suggestions", "recommend", "recommendation",
        "recommendations", "improvement", "improvements",
        "expectation", "expectations"
    ]
    sentiment_keywords = [
        "sentiment", "sentiments", "overall", "general",
        "overview", "analysis", "summary", "both",
        "feedback", "feedbacks"
    ]

    query_intent = "sentiment"
    if any(phrase in query_lower for phrase in complaint_keywords):
        query_intent = "complaint"
    elif any(phrase in query_lower for phrase in positive_keywords):
        query_intent = "positive"
    elif any(phrase in query_lower for phrase in suggestion_keywords):
        query_intent = "suggestion"
    elif any(word in query_lower for word in sentiment_keywords):
        query_intent = "sentiment"

    category_filter = SUGGESTION_CATEGORY if query_intent == "suggestion" else None

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)

    if not all_years:
        if re.search(r'\byear[- ]over[- ]year\b|\byoy\b', query_lower):
            latest = get_latest_year_from_index(index)
            try:
                all_years = [str(int(latest) - 1), latest]
            except ValueError:
                pass
        elif re.search(r'\blast\s+year\b', query_lower):
            latest = get_latest_year_from_index(index)
            try:
                all_years = [str(int(latest) - 1)]
            except ValueError:
                pass
        elif re.search(r'\bthis\s+year\b|\bcurrent\s+year\b', query_lower):
            all_years = [get_latest_year_from_index(index)]
        detected_year = all_years[0] if all_years else None

    try:
        query_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[user_query],
            parameters={"input_type": "query", "dimension": EMBEDDING_DIMENSION}
        )
        query_vector = query_response[0].values
    except Exception as e:
        return {"kind": "no_data", "reply": f"Query embedding failed: {e}"}

    active_product = detect_product_known(query_lower)
    if not active_product:
        active_product = detect_product_dynamic(query_lower, index, pc)
    active_crop = detect_crop(query_lower)
    if active_product and active_crop and active_product.lower() == active_crop.lower():
        active_product = None

    retrieval_vector = query_vector
    retrieval_top_k = 100
    subject_for_embed = " ".join(filter(None, [active_crop, active_product]))
    if subject_for_embed:
        try:
            product_embed_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[f"{subject_for_embed} product feedback sentiment complaints praise"],
                parameters={"input_type": "query", "dimension": EMBEDDING_DIMENSION}
            )
            retrieval_vector = product_embed_response[0].values
            retrieval_top_k = 300
        except Exception:
            retrieval_vector = query_vector

    # ── Deterministic ranking path ──
    if aggregation_dimension:
        return _resolve_ranking(aggregation_dimension, query_intent, category_filter, detected_month, detected_year, index)

    # ── Monthly trend path ──
    if wants_trend:
        return _resolve_trend(query_intent, category_filter, active_crop, active_product, index)

    # ── Comparison auto-detection ──
    periods = build_comparison_periods(all_months, all_years, all_weeks, index)

    if periods:
        period_results = []
        for label, m, y, w in periods:
            p_pos, p_neg, p_neut = query_pinecone_for_timeframe(
                index, retrieval_vector, m, y, w, query_intent, top_k=retrieval_top_k, category_filter=category_filter
            )
            if active_product:
                p_pos = filter_bullets_by_product(p_pos, active_product)
                p_neg = filter_bullets_by_product(p_neg, active_product)
                p_neut = filter_bullets_by_product(p_neut, active_product)
            if active_crop:
                p_pos = filter_bullets_by_crop(p_pos, active_crop)
                p_neg = filter_bullets_by_crop(p_neg, active_crop)
                p_neut = filter_bullets_by_crop(p_neut, active_crop)

            MAX_BULLETS = 12
            period_results.append((label, p_pos[:MAX_BULLETS], p_neg[:MAX_BULLETS], p_neut[:MAX_BULLETS]))

        total_found = sum(len(pp) + len(pn) + len(pu) for _, pp, pn, pu in period_results)
        timeframe_label = " vs ".join(p[0] for p in periods)
        positive_bullets = negative_bullets = neutral_bullets = None
        target_year = None

    else:
        target_year = detected_year
        fallback_triggered = False
        latest_index_year = None

        if detected_month and not target_year:
            latest_index_year = get_latest_year_from_index(index)
            target_year = latest_index_year

            pos, neg, neut = query_pinecone_for_timeframe(
                index, retrieval_vector, detected_month, target_year, detected_week, query_intent, top_k=retrieval_top_k, category_filter=category_filter
            )

            if (len(pos) + len(neg) + len(neut)) == 0:
                try:
                    fallback_year = str(int(latest_index_year) - 1)
                    pos_fb, neg_fb, neut_fb = query_pinecone_for_timeframe(
                        index, retrieval_vector, detected_month, fallback_year, detected_week, query_intent, top_k=retrieval_top_k, category_filter=category_filter
                    )
                    if (len(pos_fb) + len(neg_fb) + len(neut_fb)) > 0:
                        target_year = fallback_year
                        fallback_triggered = True
                except ValueError:
                    pass

        if wants_last_week and not detected_week:
            resolved_week = get_max_week_label(index, detected_month, target_year)
            if resolved_week:
                detected_week = resolved_week

        positive_bullets, negative_bullets, neutral_bullets = query_pinecone_for_timeframe(
            index, retrieval_vector, detected_month, target_year, detected_week, query_intent, top_k=retrieval_top_k, category_filter=category_filter
        )

        if active_product:
            positive_bullets = filter_bullets_by_product(positive_bullets, active_product)
            negative_bullets = filter_bullets_by_product(negative_bullets, active_product)
            neutral_bullets = filter_bullets_by_product(neutral_bullets, active_product)
        if active_crop:
            positive_bullets = filter_bullets_by_crop(positive_bullets, active_crop)
            negative_bullets = filter_bullets_by_crop(negative_bullets, active_crop)
            neutral_bullets = filter_bullets_by_crop(neutral_bullets, active_crop)

        total_found = len(positive_bullets) + len(negative_bullets) + len(neutral_bullets)

        timeframe_parts = []
        if detected_week:
            week_part = detected_week if "week" in detected_week.lower() else f"{detected_week} Week"
            timeframe_parts.append(week_part)
        if detected_month:
            timeframe_parts.append(f"of {detected_month}" if detected_week else detected_month)
        if target_year:
            timeframe_parts.append(target_year)
        timeframe_label = " ".join(timeframe_parts) or "the requested period"
        period_results = None

    header = build_header(query_intent, timeframe_label, active_product, periods, active_crop)
    badge = build_intent_badge(query_intent, active_product, periods, active_crop)
    subject_label = build_subject_label(active_product, active_crop)

    if total_found == 0:
        if periods:
            subject = f" for {subject_label}" if subject_label else ""
            reply = f"{badge}\n\n{header}No data found{subject} for the compared periods: {timeframe_label}."
        elif subject_label:
            suffix = f" in {timeframe_label}" if timeframe_label != "the requested period" else " in the ingested dataset"
            reply = f"{badge}\n\n{header}No data found for '{subject_label}'{suffix}."
        elif detected_month or detected_year or detected_week:
            reply = f"{badge}\n\n{header}No data found for {timeframe_label} in the ingested dataset."
        else:
            reply = (
                "I cannot generate this response. "
                "I am strictly locked to analyzed dataset metrics "
                "and cannot find relevant information for this query."
            )
        return {"kind": "no_data", "reply": reply}

    MAX_BULLETS = 12

    if periods:
        context_parts = []
        actual_point_count = 0
        for label, pos, neg, neut in period_results:
            section_lines = [f"=== {label} ==="]
            if query_intent == "complaint":
                pos = []
            elif query_intent == "positive":
                neg = []
                neut = []
            elif query_intent == "suggestion":
                pos = []
                neg = []
            if pos:
                section_lines.append("POSITIVE DATA:\n" + "\n".join(pos))
            if neg:
                section_lines.append("NEGATIVE DATA:\n" + "\n".join(neg))
            if neut:
                section_lines.append("OTHER DATA:\n" + "\n".join(neut))
            actual_point_count += len(pos) + len(neg) + len(neut)
            context_parts.append("\n".join(section_lines))
        combined_context = "\n\n".join(context_parts)
    else:
        if query_intent == "complaint":
            positive_bullets = []
            negative_bullets = negative_bullets[:MAX_BULLETS]
            neutral_bullets = neutral_bullets[:MAX_BULLETS]
        elif query_intent == "positive":
            positive_bullets = positive_bullets[:MAX_BULLETS]
            negative_bullets = []
            neutral_bullets = []
        elif query_intent == "suggestion":
            positive_bullets = []
            negative_bullets = []
            neutral_bullets = neutral_bullets[:MAX_BULLETS]
        else:
            positive_bullets = positive_bullets[:MAX_BULLETS]
            negative_bullets = negative_bullets[:MAX_BULLETS]
            neutral_bullets = neutral_bullets[:MAX_BULLETS]

        actual_point_count = len(positive_bullets) + len(negative_bullets) + len(neutral_bullets)

        context_parts = []
        if positive_bullets:
            context_parts.append("POSITIVE DATA:\n" + "\n".join(positive_bullets))
        if negative_bullets:
            context_parts.append("NEGATIVE DATA:\n" + "\n".join(negative_bullets))
        if neutral_bullets:
            context_parts.append("OTHER DATA:\n" + "\n".join(neutral_bullets))
        combined_context = "\n\n".join(context_parts)

    system_prompt = build_system_prompt(
        query_intent, timeframe_label, explicit_list_format, active_product, periods,
        active_crop=active_crop, output_format=output_format, wants_products_only=wants_products_only
    )
    user_prompt = (
        f"Timeframe: {timeframe_label}\n\n"
        f"Data Context ({actual_point_count} distinct data point{'s' if actual_point_count != 1 else ''} total — "
        f"do not exceed this number):\n{combined_context}\n\n"
        f"User Query: {user_query}"
    )
    response_token_budget = 900 if output_format in ("exec_summary", "table", "ppt") else 500

    return {
        "kind": "normal",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "header": header,
        "badge": badge,
        "timeframe_label": timeframe_label,
        "subject_label": subject_label,
        "query_intent": query_intent,
        "output_format": output_format,
        "response_token_budget": response_token_budget,
        "periods": periods,
        "period_results": period_results,
        "positive_bullets": positive_bullets,
        "negative_bullets": negative_bullets,
        "neutral_bullets": neutral_bullets,
        "actual_point_count": actual_point_count,
    }


def call_groq(system_prompt: str, user_prompt: str, groq_api_key: str, max_tokens: int = 500):
    """ Thin convenience wrapper — returns the Groq stream iterator. Each UI decides how to consume it: Streamlit updates st.empty() per chunk, a FastAPI endpoint would forward chunks as SSE. """
    client = Groq(api_key=groq_api_key)
    return client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        stream=True
    )


def finalize_normal_response(state: dict, full_response: str) -> dict:
    """ Second phase for kind="normal": once the caller has streamed the full LLM response text, compute the chart, export rows, and ready-to-serve CSV/Excel/PPTX bytes. Returns {"final_reply", "chart", "downloads"}. """
    periods = state["periods"]
    period_results = state["period_results"]
    positive_bullets = state["positive_bullets"]
    negative_bullets = state["negative_bullets"]
    neutral_bullets = state["neutral_bullets"]
    timeframe_label = state["timeframe_label"]
    header = state["header"]
    badge = state["badge"]
    subject_label = state["subject_label"]
    query_intent = state["query_intent"]
    actual_point_count = state["actual_point_count"]

    export_rows = []
    if periods:
        for label, pp, pn, pu in period_results:
            for b in pp:
                export_rows.append({"Period": label, "Sentiment": "Positive", "Feedback": b})
            for b in pn:
                export_rows.append({"Period": label, "Sentiment": "Negative", "Feedback": b})
            for b in pu:
                export_rows.append({"Period": label, "Sentiment": "Other", "Feedback": b})
    else:
        for b in positive_bullets:
            export_rows.append({"Period": timeframe_label, "Sentiment": "Positive", "Feedback": b})
        for b in negative_bullets:
            export_rows.append({"Period": timeframe_label, "Sentiment": "Negative", "Feedback": b})
        for b in neutral_bullets:
            export_rows.append({"Period": timeframe_label, "Sentiment": "Other", "Feedback": b})

    if periods:
        chart_labels = [label for label, *_ in period_results]
        chart_values = [len(pp) + len(pn) + len(pu) for _, pp, pn, pu in period_results]
        chart_title = "Total Data Points by Period"
    else:
        chart_labels = ["Positive", "Negative", "Other"]
        chart_values = [len(positive_bullets), len(negative_bullets), len(neutral_bullets)]
        chart_title = "Sentiment Breakdown"

    kpis = {
        "Total data points": actual_point_count,
        "Positive": sum(len(pp) for _, pp, _, _ in period_results) if periods else len(positive_bullets),
        "Negative": sum(len(pn) for _, _, pn, _ in period_results) if periods else len(negative_bullets),
        "Other": sum(len(pu) for _, _, _, pu in period_results) if periods else len(neutral_bullets),
    }
    summary_points = split_into_points(full_response, max_points=6)

    pptx_bytes = build_pptx_report(
        title=subject_label or f"{query_intent.title()} Analysis",
        subtitle=timeframe_label,
        exec_summary_lines=summary_points[:4] or ["No summary available."],
        kpis=kpis,
        chart_title=chart_title,
        chart_labels=chart_labels,
        chart_values=chart_values,
        chart_type="column",
        table_headers=["Sentiment", "Feedback"],
        table_rows=[(row["Sentiment"], row["Feedback"]) for row in export_rows],
        insights=summary_points[:4],
        recommendations=summary_points[4:6],
    )

    downloads = {
        "csv": build_csv_bytes([{"Label": l, "Value": v} for l, v in zip(chart_labels, chart_values)]),
        "excel": build_excel_bytes(export_rows) if export_rows else None,
        "pptx": pptx_bytes,
    }

    final_reply = badge + "\n\n" + header + full_response

    return {
        "final_reply": final_reply,
        "chart": {"type": "bar", "title": chart_title, "labels": chart_labels, "values": chart_values},
        "downloads": downloads,
        "kpis": kpis,
    }


def _resolve_ranking(aggregation_dimension, query_intent, category_filter, detected_month, detected_year, index) -> dict:
    agg_filter = {}
    if detected_month:
        agg_filter["month"] = {"$eq": detected_month}
    if detected_year:
        agg_filter["year"] = {"$eq": detected_year}
    if query_intent == "positive":
        agg_filter["sentiment"] = {"$eq": "positive"}
    elif query_intent == "complaint":
        agg_filter["sentiment"] = {"$eq": "negative"}
    elif category_filter:
        agg_filter["category"] = {"$eq": category_filter}

    agg_matches = fetch_matches_for_aggregation(index, agg_filter)
    field = "crop" if aggregation_dimension == "crop" else "products"
    ranking = rank_by_field(agg_matches, field, top_n=10)

    badge = f"📊 {aggregation_dimension.title()} Ranking"
    scope_bits = []
    if detected_month:
        scope_bits.append(detected_month)
    if detected_year:
        scope_bits.append(detected_year)
    scope_label = " ".join(scope_bits) if scope_bits else "all available data"
    header = f"📊 {aggregation_dimension.title()}-wise Ranking ({scope_label}):\n\n"

    if not ranking:
        reply = (
            f"{header}No {aggregation_dimension} tags were found in the matched "
            f"records for {scope_label} — nothing to rank."
        )
        return {"kind": "ranking", "reply": reply, "badge": badge, "chart": None, "downloads": None}

    table_lines = ["| Rank | " + aggregation_dimension.title() + " | Mentions |", "|---|---|---|"]
    for i, (name, count) in enumerate(ranking, start=1):
        table_lines.append(f"| {i} | {name} | {count} |")
    top_name, top_count = ranking[0]
    reply = (
        f"{header}"
        f"Based on {len(agg_matches)} matched records, **{top_name}** ranks highest "
        f"with {top_count} mention{'s' if top_count != 1 else ''}.\n\n"
        + "\n".join(table_lines)
    )

    labels = [n for n, _ in ranking]
    values = [c for _, c in ranking]

    pptx_bytes = build_pptx_report(
        title=f"{aggregation_dimension.title()}-wise Ranking",
        subtitle=scope_label,
        exec_summary_lines=[f"Based on {len(agg_matches)} matched records, {top_name} ranks highest with {top_count} mentions."],
        kpis={"Total matched records": len(agg_matches), "Top " + aggregation_dimension: top_name, "Top mentions": top_count},
        chart_title=f"{aggregation_dimension.title()} Mentions", chart_labels=labels, chart_values=values,
        chart_type="column",
        table_headers=["Rank", aggregation_dimension.title(), "Mentions"],
        table_rows=[(i, n, c) for i, (n, c) in enumerate(ranking, start=1)],
        insights=[f"{n} — {c} mentions" for n, c in ranking[:5]],
    )
    downloads = {
        "csv": build_csv_bytes([{aggregation_dimension.title(): n, "Mentions": c} for n, c in ranking]),
        "excel": build_excel_bytes([{aggregation_dimension.title(): n, "Mentions": c} for n, c in ranking]),
        "pptx": pptx_bytes,
    }

    return {
        "kind": "ranking",
        "reply": reply,
        "badge": badge,
        "chart": {"type": "bar", "title": f"{aggregation_dimension.title()} Mentions", "labels": labels, "values": values},
        "downloads": downloads,
    }


def _resolve_trend(query_intent, category_filter, active_crop, active_product, index) -> dict:
    trend_filter = {}
    if query_intent == "positive":
        trend_filter["sentiment"] = {"$eq": "positive"}
    elif query_intent == "complaint":
        trend_filter["sentiment"] = {"$eq": "negative"}
    elif category_filter:
        trend_filter["category"] = {"$eq": category_filter}

    trend_matches = fetch_matches_for_aggregation(index, trend_filter)
    if active_crop:
        trend_matches = [m for m in trend_matches if active_crop.lower() in str(m.get("metadata", {}).get("value", "")).lower()]
    if active_product:
        trend_matches = [m for m in trend_matches if active_product.lower() in str(m.get("metadata", {}).get("value", "")).lower()]

    monthly_counts = compute_monthly_trend(trend_matches)
    growth_series = compute_growth_series(monthly_counts)

    subject_label = build_subject_label(active_product, active_crop)
    subject_bit = f" for {subject_label}" if subject_label else ""
    header = f"📈 Monthly Trend Analysis{subject_bit}:\n\n"

    if not monthly_counts:
        reply = f"{header}No dated records were found{subject_bit} to build a monthly trend from."
        return {"kind": "no_data", "reply": reply}

    highest = max(monthly_counts, key=lambda x: x[1])
    lowest = min(monthly_counts, key=lambda x: x[1])
    table_lines = ["| Month | Count | MoM Growth |", "|---|---|---|"]
    for (label, count), growth in zip(monthly_counts, growth_series):
        growth_str = "—" if growth is None else f"{'+' if growth >= 0 else ''}{growth}%"
        table_lines.append(f"| {label} | {count} | {growth_str} |")

    reply = (
        f"{header}"
        f"Highest month: **{highest[0]}** ({highest[1]} records). "
        f"Lowest month: **{lowest[0]}** ({lowest[1]} records).\n\n"
        + "\n".join(table_lines)
    )

    labels = [l for l, _ in monthly_counts]
    values = [c for _, c in monthly_counts]

    pptx_bytes = build_pptx_report(
        title=f"Monthly Trend Analysis{subject_bit}",
        subtitle=f"{monthly_counts[0][0]} – {monthly_counts[-1][0]}",
        exec_summary_lines=[
            f"Highest month: {highest[0]} with {highest[1]} records.",
            f"Lowest month: {lowest[0]} with {lowest[1]} records.",
        ],
        kpis={"Total records": sum(values), "Highest month": f"{highest[0]} ({highest[1]})",
              "Lowest month": f"{lowest[0]} ({lowest[1]})"},
        chart_title="Monthly Trend", chart_labels=labels, chart_values=values, chart_type="line",
        table_headers=["Month", "Count", "MoM Growth %"],
        table_rows=[(l, c, ("—" if g is None else f"{g}%")) for (l, c), g in zip(monthly_counts, growth_series)],
        insights=[f"{highest[0]} was the strongest month ({highest[1]} records).",
                  f"{lowest[0]} was the weakest month ({lowest[1]} records)."],
    )
    downloads = {
        "csv": build_csv_bytes([{"Month": l, "Count": c, "MoM Growth %": g} for (l, c), g in zip(monthly_counts, growth_series)]),
        "excel": build_excel_bytes([{"Month": l, "Count": c, "MoM Growth %": g} for (l, c), g in zip(monthly_counts, growth_series)]),
        "pptx": pptx_bytes,
    }

    return {
        "kind": "trend",
        "reply": reply,
        "badge": "📈 Monthly Trend",
        "chart": {"type": "line", "title": "Monthly Trend", "labels": labels, "values": values},
        "downloads": downloads,
    }
