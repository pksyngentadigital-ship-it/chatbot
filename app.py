import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os

# ── APP BUILD MARKER ── (bump this string whenever the file is regenerated,
# so it's easy to confirm in the sidebar/logs which version is deployed)
APP_BUILD = "2026-07-15-v8 (VoG capability expansion: layout-aware ingestion, crop/product ranking, suggestions, output formats)"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", None)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PINECONE_INDEX_NAME = "chatbot"
EMBEDDING_DIMENSION = 384
GROQ_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Weekly Sentiment RAG Engine", page_icon="🌾", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# UI STYLING (cosmetic only — agriculture theme)
# ==========================================
st.markdown(""" <style> .stApp { background: linear-gradient(180deg, #f3f9f1 0%, #eaf4e6 100%); } section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1b3a24 0%, #0f2417 100%); } section[data-testid="stSidebar"] * { color: #eef7ec !important; } section[data-testid="stSidebar"] input { color: #111 !important; } section[data-testid="stSidebar"] button { background-color: #2e7d32 !important; border: 1px solid #256029 !important; border-radius: 8px !important; } section[data-testid="stSidebar"] button, section[data-testid="stSidebar"] button p, section[data-testid="stSidebar"] button span, section[data-testid="stSidebar"] button div { color: #ffffff !important; } section[data-testid="stSidebar"] button:hover { background-color: #256029 !important; border-color: #1b3a24 !important; } section[data-testid="stSidebar"] button:hover, section[data-testid="stSidebar"] button:hover p, section[data-testid="stSidebar"] button:hover span, section[data-testid="stSidebar"] button:hover div { color: #ffffff !important; } .hero-title { font-size: 2.15rem; font-weight: 800; background: linear-gradient(90deg, #2e7d32, #558b2f, #33691e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.1rem; } .hero-subtitle { color: #4b5d4e; font-size: 0.97rem; margin-bottom: 1.2rem; } div[data-testid="stChatMessage"] { border-radius: 16px; padding: 0.7rem 1.1rem; margin-bottom: 0.6rem; box-shadow: 0 1px 5px rgba(46, 125, 50, 0.10); background: #f2f9f2; border: 1px solid #e2f0e2; } div[data-testid="stChatMessage"] ul { list-style: none; padding-left: 0.1rem; margin-top: 0.4rem; } div[data-testid="stChatMessage"] li { position: relative; padding-left: 1.5rem; margin-bottom: 0.45rem; line-height: 1.45; } div[data-testid="stChatMessage"] li::before { content: "🌱"; position: absolute; left: 0; top: 0; } .intent-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.55rem; } .badge-positive { background: #dff5df; color: #256029; } .badge-complaint { background: #fdeaea; color: #9c3b3b; } .badge-sentiment { background: #e3f1e6; color: #2e5d34; } .badge-comparison { background: #eee3f9; color: #5b3a94; } .badge-product { background: #fff3d6; color: #8a5a00; } .badge-suggestion { background: #e6e6fa; color: #4b3f8a; } .badge-ranking { background: #fde2c8; color: #8a4b00; } div[data-testid="stChatInput"] textarea { border-radius: 12px !important; } h1, .hero-title { display: flex; align-items: center; gap: 0.4rem; } </style> """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR: CREDENTIALS & CONFIG
# ==========================================
with st.sidebar:
    st.header("⚙️ System Credentials")
    st.markdown("---")
    st.subheader("🔑 Admin Panel")
    if not st.session_state.authenticated:
        admin_password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if admin_password == "admin123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid Credentials")
    else:
        st.write("🟢 Authorized Mode")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

    st.markdown("---")
    st.caption(f"Build: {APP_BUILD}")

# ==========================================
# CONSTANTS & DICTIONARIES
# ==========================================

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
    "business", "monthly", "yearly", "annual", "annually", "breakdown"
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
    "two", "three", "four", "five", "recent", "latest", "last", "current"
} | set(MONTH_MAP.keys())

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


# ==========================================
# NEW UTILITIES: multi-value extraction, product
# detection, and comparison support
# ==========================================

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
    """Small colored pill shown above the streamed answer — purely cosmetic."""
    subject_label = build_subject_label(active_product, active_crop)
    if periods:
        return '<span class="intent-badge badge-comparison">🔀 Comparison</span>'
    if subject_label:
        return f'<span class="intent-badge badge-product">🏷️ {subject_label}</span>'
    if query_intent == "complaint":
        return '<span class="intent-badge badge-complaint">🐛 Complaints</span>'
    if query_intent == "positive":
        return '<span class="intent-badge badge-positive">🌻 Positive</span>'
    if query_intent == "suggestion":
        return '<span class="intent-badge badge-suggestion">💡 Suggestions</span>'
    return '<span class="intent-badge badge-sentiment">🌾 Sentiment Overview</span>'


def build_system_prompt(query_intent, timeframe_label, explicit_list_format, active_product, periods, active_crop=None, output_format=None):
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
        f"{comparison_clause}"
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
# ADMIN: EXCEL INGESTION
# ==========================================
if st.session_state.authenticated:
    st.title("📥 Dataset Pipeline Ingestion Panel")
    uploaded_file = st.file_uploader("Upload Master Performance Log (.xlsx)", type=["xlsx"])

    if uploaded_file and PINECONE_API_KEY:
        if st.button("Process Sheets & Map Matrix"):
            progress_bar = st.progress(0)
            status_text  = st.empty()

            with st.spinner("Executing server-side matrix mapping..."):
                try:
                    file_bytes  = BytesIO(uploaded_file.read())
                    excel_file  = pd.ExcelFile(file_bytes)
                    all_sheets  = excel_file.sheet_names

                    pc    = Pinecone(api_key=PINECONE_API_KEY)
                    index = pc.Index(PINECONE_INDEX_NAME)

                    payload_batch             = []
                    text_inputs_for_embedding = []
                    discovered_data_summary   = {}

                    def make_metadata_payload(inferred_year, row_month, week_label, category, bullet):
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

                    for sheet_name in all_sheets:
                        sheet_clean   = sheet_name.strip()
                        inferred_year = infer_year_for_sheet(sheet_clean, all_sheets)
                        if not inferred_year:
                            continue

                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

                        cat_col = find_category_column(df.columns)

                        if cat_col:
                            # ── LAYOUT A: categories are ROWS, weeks are COLUMNS ──
                            # (this is how every explicitly-year-labeled sheet, e.g.
                            # "Jan till Dec 2024", is structured)
                            week_cols = [c for c in df.columns if 'week' in c.lower()]

                            for idx, row in df.iterrows():
                                raw_category = row.get(cat_col, None)
                                category     = normalize_category(raw_category)

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
                                    stat_key  = f"{row_month} {inferred_year}"
                                    discovered_data_summary[stat_key] = (
                                        discovered_data_summary.get(stat_key, 0) + len(bullets)
                                    )

                                    for b_idx, bullet in enumerate(bullets):
                                        context_chunk, metadata_payload = make_metadata_payload(
                                            inferred_year, row_month, col, category, bullet
                                        )
                                        clean_cat   = re.sub(r'[^a-zA-Z0-9]', '', category.replace(' ', '_'))
                                        clean_col   = re.sub(r'[^a-zA-Z0-9]', '', col.replace(' ', '_'))
                                        clean_sheet = re.sub(r'[^a-zA-Z0-9]', '', sheet_clean.replace(' ', '_'))
                                        vector_id   = f"v_{clean_sheet}{clean_cat}{clean_col}{idx}{b_idx}"

                                        payload_batch.append({"id": vector_id, "metadata": metadata_payload})
                                        text_inputs_for_embedding.append(context_chunk)

                        else:
                            # ── LAYOUT B: Month/Week are ROW values, categories are
                            # COLUMN headers (e.g. the legacy undated sheets). Locate
                            # the real header row (the one containing a "Month" cell),
                            # then forward-fill Month/Week down merged blocks. ──
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
                            week_col_idx  = next((i for i, v in col_map.items() if v.strip().lower() == 'week'), None)
                            category_cols = {i: v for i, v in col_map.items() if i not in (month_col_idx, week_col_idx)}
                            if month_col_idx is None or not category_cols:
                                continue

                            current_month = None
                            current_week  = None
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
                                        context_chunk, metadata_payload = make_metadata_payload(
                                            inferred_year, current_month, week_label, category, bullet
                                        )
                                        clean_cat   = re.sub(r'[^a-zA-Z0-9]', '', category.replace(' ', '_'))
                                        clean_week  = re.sub(r'[^a-zA-Z0-9]', '', week_label.replace(' ', '_'))
                                        clean_sheet = re.sub(r'[^a-zA-Z0-9]', '', sheet_clean.replace(' ', '_'))
                                        vector_id   = f"v_{clean_sheet}{clean_cat}{clean_week}{r}{b_idx}"

                                        payload_batch.append({"id": vector_id, "metadata": metadata_payload})
                                        text_inputs_for_embedding.append(context_chunk)

                    total_records = len(payload_batch)
                    if total_records == 0:
                        st.warning("No records found.")
                        st.stop()

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
                            "id":       item["id"],
                            "values":   all_vectors[i],
                            "metadata": item["metadata"]
                        })
                        if len(upsert_buffer) >= 50:
                            index.upsert(vectors=upsert_buffer)
                            upsert_buffer = []

                    if upsert_buffer:
                        index.upsert(vectors=upsert_buffer)

                    st.success(f"🎉 Pipeline complete! Ingested {total_records} records.")

                except Exception as e:
                    st.error(f"Inbound data ingestion pipe error: {e}")

# ==========================================
# PUBLIC CHAT INTERFACE
# ==========================================
st.markdown('<div class="hero-title">🌾 Strategic Enterprise Performance Analyzer 🌱</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask about sentiment 🌾, complaints 🐛, positive feedback 🌻, a specific '
    'product 🏷️, or compare weeks / months / years 🔀.</div>',
    unsafe_allow_html=True
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

user_query = st.chat_input("Ask about sentiment, a product, or compare periods...")

if user_query and user_query.strip():
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # ── STRICT TOPIC GUARDRAIL ──
    allowed_keywords = set([
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
    query_words = re.findall(r'\b\w+\b', user_query.lower())
    is_relevant = any(word in allowed_keywords for word in query_words)

    if not is_relevant:
        reply = (
            "I cannot generate this response. "
            "I am strictly locked to analyzed dataset metrics "
            "and cannot find relevant information for this query."
        )
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.stop()

    if not PINECONE_API_KEY:
        with st.chat_message("assistant"):
            st.markdown("🤖 Execution Halted: Pinecone API key is not configured.")
        st.stop()

    with st.spinner("Searching and aggregating matching historical data records..."):

        query_lower = user_query.lower()

        # ── Multi-value extraction (powers auto comparison detection) ──
        all_months = extract_all_months(query_lower)
        all_years  = extract_all_years(query_lower)
        all_weeks  = extract_all_weeks(query_lower)

        detected_month = all_months[0] if all_months else None
        detected_year  = all_years[0] if all_years else None
        detected_week  = all_weeks[0] if all_weeks else None

        # ── "last / latest / most recent week" → resolve to the real latest
        # week label in the data (only when no explicit ordinal week like
        # "2nd week" was already given) ──
        wants_last_week = bool(re.search(r'\b(last|latest|most recent|recent)\s+week\b', query_lower))

        # ── Explicit "list it out" detection → bullet formatting ──
        explicit_list_format = bool(re.search(r'\blist(ed|ing)?\b|\bbullets?\b|\bbullet\s*points?\b', query_lower))

        # ── Output format the user explicitly asked for (table / exec summary / PPT / chart / excel) ──
        output_format = detect_output_format(query_lower)

        # ── Deterministic ranking request ("which crop/product generated the highest...") ──
        aggregation_dimension = detect_aggregation_request(query_lower)

        # ==========================================
        # INTENT DETECTION
        # ==========================================
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

        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # ── "year-over-year" / "YoY" / "last year" / "this year" phrasing →
        # resolve to concrete years when the user didn't name them explicitly ──
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
            st.error(f"Query embedding failed: {e}")
            st.stop()

        # ── Product & crop detection: curated list first, dynamic probe fallback (product only) ──
        active_product = detect_product_known(query_lower)
        if not active_product:
            active_product = detect_product_dynamic(query_lower, index, pc)
        active_crop = detect_crop(query_lower)
        if active_product and active_crop and active_product.lower() == active_crop.lower():
            active_product = None

        # ── Retrieval vector: once a product/crop is known, search using a
        # focused embedding instead of the raw user phrasing. This makes
        # "tell me about Axial" behave the same as "give me feedback of
        # Axial" — retrieval no longer depends on how the question happens
        # to be worded. ──
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

        # ── Deterministic ranking path ("which crop/product generated the
        # highest number of complaints") — computed by exact counting over
        # metadata tags, never left to the LLM to eyeball. Bypasses the
        # normal retrieval/LLM flow entirely. ──
        if aggregation_dimension:
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

            badge = f'<span class="intent-badge badge-ranking">📊 {aggregation_dimension.title()} Ranking</span>'
            scope_bits = []
            if detected_month:
                scope_bits.append(detected_month)
            if detected_year:
                scope_bits.append(detected_year)
            scope_label = " ".join(scope_bits) if scope_bits else "all available data"
            header = f"📊 {aggregation_dimension.title()}-wise Ranking ({scope_label}):\n\n"

            if not ranking:
                reply = (
                    f"{badge}\n\n{header}No {aggregation_dimension} tags were found in the matched "
                    f"records for {scope_label} — nothing to rank."
                )
            else:
                table_lines = ["| Rank | " + aggregation_dimension.title() + " | Mentions |", "|---|---|---|"]
                for i, (name, count) in enumerate(ranking, start=1):
                    table_lines.append(f"| {i} | {name} | {count} |")
                top_name, top_count = ranking[0]
                reply = (
                    f"{badge}\n\n{header}"
                    f"Based on {len(agg_matches)} matched records, **{top_name}** ranks highest "
                    f"with {top_count} mention{'s' if top_count != 1 else ''}.\n\n"
                    + "\n".join(table_lines)
                )

            with st.chat_message("assistant"):
                st.markdown(reply, unsafe_allow_html=True)
                if ranking:
                    chart_df = pd.DataFrame(
                        {aggregation_dimension.title(): [n for n, _ in ranking], "Mentions": [c for _, c in ranking]}
                    ).set_index(aggregation_dimension.title())
                    st.bar_chart(chart_df)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.stop()

        # ── Comparison auto-detection: 2+ distinct months/years/weeks
        # mentioned is enough — no "compare" keyword required. ──
        periods = build_comparison_periods(all_months, all_years, all_weeks, index)

        if periods:
            # ── COMPARISON FLOW ──
            period_results = []
            for label, m, y, w in periods:
                p_pos, p_neg, p_neut = query_pinecone_for_timeframe(
                    index, retrieval_vector, m, y, w, query_intent, top_k=retrieval_top_k, category_filter=category_filter
                )
                if active_product:
                    p_pos  = filter_bullets_by_product(p_pos, active_product)
                    p_neg  = filter_bullets_by_product(p_neg, active_product)
                    p_neut = filter_bullets_by_product(p_neut, active_product)
                if active_crop:
                    p_pos  = filter_bullets_by_crop(p_pos, active_crop)
                    p_neg  = filter_bullets_by_crop(p_neg, active_crop)
                    p_neut = filter_bullets_by_crop(p_neut, active_crop)

                MAX_BULLETS = 12
                period_results.append((label, p_pos[:MAX_BULLETS], p_neg[:MAX_BULLETS], p_neut[:MAX_BULLETS]))

            total_found = sum(len(pp) + len(pn) + len(pu) for _, pp, pn, pu in period_results)
            timeframe_label = " vs ".join(p[0] for p in periods)
            fallback_triggered = False
            target_year = None

        else:
            # ── ORIGINAL SINGLE-PERIOD FLOW (unchanged) ──
            target_year        = detected_year
            fallback_triggered = False
            latest_index_year  = None

            if detected_month and not target_year:
                latest_index_year = get_latest_year_from_index(index)
                target_year       = latest_index_year

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
                            target_year        = fallback_year
                            fallback_triggered = True
                    except ValueError:
                        pass

            # ── Resolve "last / latest / recent week" to the real latest
            # week label present in the data for this month/year ──
            if wants_last_week and not detected_week:
                resolved_week = get_max_week_label(index, detected_month, target_year)
                if resolved_week:
                    detected_week = resolved_week

            positive_bullets, negative_bullets, neutral_bullets = query_pinecone_for_timeframe(
                index, retrieval_vector, detected_month, target_year, detected_week, query_intent, top_k=retrieval_top_k, category_filter=category_filter
            )

            # ── Product / crop filter ──
            if active_product:
                positive_bullets = filter_bullets_by_product(positive_bullets, active_product)
                negative_bullets = filter_bullets_by_product(negative_bullets, active_product)
                neutral_bullets  = filter_bullets_by_product(neutral_bullets, active_product)
            if active_crop:
                positive_bullets = filter_bullets_by_crop(positive_bullets, active_crop)
                negative_bullets = filter_bullets_by_crop(negative_bullets, active_crop)
                neutral_bullets  = filter_bullets_by_crop(neutral_bullets, active_crop)

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

    # ── Info banners for the single-period flow only ──
    if not periods:
        if fallback_triggered:
            st.info(
                f"ℹ️ No data found for {detected_month} {latest_index_year}. "
                f"Automatically falling back to {target_year}."
            )
        elif detected_month and not detected_year:
            st.info(
                f"ℹ️ Year not specified. Defaulting to the latest available dataset year: {target_year}"
            )

    header = build_header(query_intent, timeframe_label, active_product, periods, active_crop)
    badge  = build_intent_badge(query_intent, active_product, periods, active_crop)
    subject_label = build_subject_label(active_product, active_crop)

    if total_found == 0:
        if periods:
            subject = f" for {subject_label}" if subject_label else ""
            reply = (
                f"{badge}\n\n{header}"
                f"No data found{subject} for the compared periods: {timeframe_label}."
            )
        elif subject_label:
            suffix = f" in {timeframe_label}" if timeframe_label != "the requested period" else " in the ingested dataset"
            reply = f"{badge}\n\n{header}No data found for '{subject_label}'{suffix}."
        elif detected_month or detected_year or detected_week:
            reply = (
                f"{badge}\n\n{header}"
                f"No data found for {timeframe_label} in the ingested dataset."
            )
        else:
            reply = (
                "I cannot generate this response. "
                "I am strictly locked to analyzed dataset metrics "
                "and cannot find relevant information for this query."
            )
        with st.chat_message("assistant"):
            st.markdown(reply, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.stop()

    MAX_BULLETS = 12

    if periods:
        # Build labeled context blocks, one per compared period
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
            neutral_bullets  = neutral_bullets[:MAX_BULLETS]
        elif query_intent == "positive":
            positive_bullets = positive_bullets[:MAX_BULLETS]
            negative_bullets = []
            neutral_bullets  = []
        elif query_intent == "suggestion":
            positive_bullets = []
            negative_bullets = []
            neutral_bullets  = neutral_bullets[:MAX_BULLETS]
        else:
            positive_bullets = positive_bullets[:MAX_BULLETS]
            negative_bullets = negative_bullets[:MAX_BULLETS]
            neutral_bullets  = neutral_bullets[:MAX_BULLETS]

        actual_point_count = len(positive_bullets) + len(negative_bullets) + len(neutral_bullets)

        context_parts = []
        if positive_bullets:
            context_parts.append("POSITIVE DATA:\n" + "\n".join(positive_bullets))
        if negative_bullets:
            context_parts.append("NEGATIVE DATA:\n" + "\n".join(negative_bullets))
        if neutral_bullets:
            context_parts.append("OTHER DATA:\n"    + "\n".join(neutral_bullets))

        combined_context = "\n\n".join(context_parts)

    system_prompt = build_system_prompt(
        query_intent, timeframe_label, explicit_list_format, active_product, periods,
        active_crop=active_crop, output_format=output_format
    )

    user_prompt = (
        f"Timeframe: {timeframe_label}\n\n"
        f"Data Context ({actual_point_count} distinct data point{'s' if actual_point_count != 1 else ''} total — "
        f"do not exceed this number):\n{combined_context}\n\n"
        f"User Query: {user_query}"
    )

    # ── Stream response with Groq ──
    with st.chat_message("assistant"):
        st.markdown(badge, unsafe_allow_html=True)
        stream_box    = st.empty()
        full_response = ""

        try:
            groq_client = Groq(api_key=GROQ_API_KEY)

            response_token_budget = 900 if output_format in ("exec_summary", "table", "ppt") else 500

            stream = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=response_token_budget,
                stream=True
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_response += token
                stream_box.markdown(header + full_response + "▌")

            stream_box.markdown(header + full_response)

        except Exception as e:
            full_response = f"Operational Processing Error: {e}"
            stream_box.markdown(header + full_response)

        # ── "chart" output format: supplement the text answer with a
        # sentiment-breakdown bar chart built from the actual retrieved data ──
        if output_format == "chart":
            if periods:
                chart_df = pd.DataFrame({
                    label: {"Positive": len(pp), "Negative": len(pn), "Other": len(pu)}
                    for label, pp, pn, pu in period_results
                }).T
            else:
                chart_df = pd.DataFrame(
                    {"Count": [len(positive_bullets), len(negative_bullets), len(neutral_bullets)]},
                    index=["Positive", "Negative", "Other"]
                )
            st.bar_chart(chart_df)

        # ── "excel"/"export" output format: offer the retrieved data points
        # as a real downloadable .xlsx (built from the same grounded data
        # fed to the LLM, so nothing here can be fabricated) ──
        if output_format == "excel":
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

            if export_rows:
                export_buffer = BytesIO()
                pd.DataFrame(export_rows).to_excel(export_buffer, index=False, engine="openpyxl")
                st.download_button(
                    "⬇️ Download as Excel",
                    data=export_buffer.getvalue(),
                    file_name="vog_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    final_reply = badge + "\n\n" + header + full_response
    st.session_state.chat_history.append({"role": "assistant", "content": final_reply})