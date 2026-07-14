import streamlit as st
import pandas as pd
import re
from io import BytesIO
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", None)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PINECONE_INDEX_NAME = "chatbot"
EMBEDDING_DIMENSION = 384
GROQ_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Syngenta Sentiment Engine", layout="wide", page_icon="🌱")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# GLOBAL CSS
# ==========================================
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3c5e 50%, #0d2b1f 100%);
        min-height: 100vh;
    }

    /* ── Hide default streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 900px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a3c5e 100%) !important;
        border-right: 1px solid rgba(0,166,81,0.3);
    }
    [data-testid="stSidebar"] * { color: #e8f4f8 !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(0,166,81,0.4) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 10px 14px !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #00a651, #007a3d) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        width: 100% !important;
        font-weight: 600 !important;
        padding: 10px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0,166,81,0.4) !important;
    }

    /* ── Chat container wrapper ── */
    .chat-wrapper {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px 16px 10px 16px;
        margin-bottom: 8px;
        min-height: 420px;
        max-height: 560px;
        overflow-y: auto;
    }

    /* ── User chat bubble ── */
    .bubble-user {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
        gap: 8px;
        margin: 6px 0;
    }
    .bubble-user .msg {
        background: linear-gradient(135deg, #00a651, #007a3d);
        color: white;
        padding: 11px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 72%;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 3px 12px rgba(0,166,81,0.35);
        word-wrap: break-word;
    }
    .bubble-user .avatar {
        width: 32px; height: 32px;
        background: rgba(0,166,81,0.2);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; flex-shrink: 0;
        border: 1px solid rgba(0,166,81,0.4);
    }

    /* ── Bot chat bubble ── */
    .bubble-bot {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 8px;
        margin: 6px 0;
    }
    .bubble-bot .avatar {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #1a3c5e, #0d2b3e);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; flex-shrink: 0;
        border: 1px solid rgba(0,166,81,0.3);
        margin-top: 2px;
    }
    .bubble-bot .msg {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(8px);
        color: #e8f4f8;
        padding: 11px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 78%;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 3px 12px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        word-wrap: break-word;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1.5px solid rgba(0,166,81,0.5) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: white !important;
        font-size: 14px !important;
    }
    [data-testid="stChatInput"] button {
        background: #00a651 !important;
        border-radius: 10px !important;
    }

    /* ── Page title ── */
    .page-title {
        text-align: center;
        padding: 6px 0 14px 0;
    }
    .page-title h1 {
        font-size: 1.7rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .page-title p {
        color: rgba(255,255,255,0.5);
        font-size: 13px;
        margin: 4px 0 0 0;
    }

    /* ── Admin panel card ── */
    .admin-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0,166,81,0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .admin-card h2 {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .admin-card p {
        color: rgba(255,255,255,0.45);
        font-size: 13px;
        margin: 0 0 22px 0;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: rgba(0,166,81,0.05) !important;
        border: 2px dashed rgba(0,166,81,0.4) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,166,81,0.8) !important;
        background: rgba(0,166,81,0.08) !important;
    }
    [data-testid="stFileUploader"] label {
        color: rgba(255,255,255,0.7) !important;
        font-size: 14px !important;
    }
    [data-testid="stFileUploader"] svg { color: #00a651 !important; }

    /* ── Process button ── */
    .stButton > button {
        background: linear-gradient(135deg, #00a651, #007a3d) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,166,81,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,166,81,0.45) !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00a651, #00d468) !important;
        border-radius: 10px !important;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }

    /* ── Info / success / warning / error boxes ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        font-size: 13px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #00a651 !important; }

    /* ── Stats row ── */
    .stat-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(0,166,81,0.2);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-card .stat-num {
        font-size: 2rem;
        font-weight: 700;
        color: #00a651;
        line-height: 1;
    }
    .stat-card .stat-label {
        font-size: 12px;
        color: rgba(255,255,255,0.5);
        margin-top: 4px;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(0,166,81,0.4);
        border-radius: 10px;
    }

    /* ── Hide streamlit chat avatars (we use custom) ── */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] { display: none !important; }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

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
    "others":                       "Others",
    "other":                        "Others"
}

POSITIVE_CATEGORIES = {"Positive Feedback"}
NEGATIVE_CATEGORIES = {"Complaint/Negative Feedback"}

EMPTY_VALUES = {
    'nan', 'none', '', 'null', '-', 'n/a', 'na',
    'not filled', 'not available', 'no data', '0', 'tbd', 'pending'
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
    direct = re.search(r'(20\d{2})', sheet_name.strip())
    if direct:
        return direct.group(0)
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

def query_pinecone_for_timeframe(index, query_vector, month, year, week, query_intent="sentiment"):
    filter_conditions = {}
    if month:
        filter_conditions["month"] = {"$eq": month}
    if year:
        filter_conditions["year"]  = {"$eq": year}
    if query_intent == "positive":
        filter_conditions["sentiment"] = {"$eq": "positive"}
    elif query_intent == "complaint":
        filter_conditions["sentiment"] = {"$eq": "negative"}

    metadata_filter = filter_conditions if filter_conditions else None
    results = index.query(
        vector=query_vector,
        top_k=100,
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

        if sentiment == "positive":
            positive_bullets.append(entry)
        elif sentiment == "negative":
            negative_bullets.append(entry)
        else:
            neutral_bullets.append(entry)

    return positive_bullets, negative_bullets, neutral_bullets

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="padding: 20px 0 10px 0; text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 6px;">🌱</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: white; letter-spacing: 0.5px;">
                Syngenta
            </div>
            <div style="font-size: 11px; color: rgba(255,255,255,0.4); margin-top: 2px;">
                Sentiment Intelligence Engine
            </div>
        </div>
        <hr style="border-color: rgba(0,166,81,0.25); margin: 10px 0 18px 0;">
    """, unsafe_allow_html=True)

    if not st.session_state.authenticated:
        st.markdown("""
            <div style="font-size: 13px; font-weight: 600;
                        color: rgba(255,255,255,0.6);
                        letter-spacing: 0.5px; margin-bottom: 10px;">
                🔑 ADMIN LOGIN
            </div>
        """, unsafe_allow_html=True)
        admin_password = st.text_input("Password", type="password", label_visibility="collapsed",
                                        placeholder="Enter admin password...")
        if st.button("Login →"):
            if admin_password == "admin123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.markdown("""
            <div style="
                background: rgba(0,166,81,0.15);
                border: 1px solid rgba(0,166,81,0.35);
                border-radius: 12px;
                padding: 12px 14px;
                margin-bottom: 14px;
            ">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:8px; height:8px; background:#00a651;
                                border-radius:50%; box-shadow: 0 0 6px #00a651;"></div>
                    <span style="color:white; font-size:13px; font-weight:600;">
                        Authorized Mode
                    </span>
                </div>
                <div style="color:rgba(255,255,255,0.45); font-size:11px; margin-top:4px; margin-left:16px;">
                    Admin access granted
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

        st.markdown("""
            <hr style="border-color: rgba(255,255,255,0.08); margin: 16px 0;">
            <div style="font-size: 11px; color: rgba(255,255,255,0.3);
                        text-align: center; padding-bottom: 8px;">
                Dataset Pipeline v2.0
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# ADMIN PANEL — INGESTION
# ==========================================
if st.session_state.authenticated:

    st.markdown("""
        <div class="admin-card">
            <h2>📥 Dataset Pipeline Ingestion</h2>
            <p>Upload your master performance log to embed and index into the vector database</p>
        </div>
    """, unsafe_allow_html=True)

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        st.markdown("""
            <div style="font-size: 13px; font-weight: 600;
                        color: rgba(255,255,255,0.6);
                        letter-spacing: 0.5px; margin-bottom: 8px;">
                📂 UPLOAD EXCEL FILE
            </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your .xlsx file here",
            type=["xlsx"],
            label_visibility="collapsed"
        )

    with col_info:
        st.markdown("""
            <div style="
                background: rgba(0,166,81,0.07);
                border: 1px solid rgba(0,166,81,0.2);
                border-radius: 14px;
                padding: 16px;
                height: 100%;
            ">
                <div style="color: rgba(255,255,255,0.7); font-size: 12px;
                            font-weight: 600; margin-bottom: 10px; letter-spacing: 0.5px;">
                    📋 REQUIREMENTS
                </div>
                <div style="color: rgba(255,255,255,0.45); font-size: 12px; line-height: 1.8;">
                    ✅ Format: <b style="color:rgba(255,255,255,0.65)">.xlsx</b><br>
                    ✅ Sheet names must include year<br>
                    ✅ Category column required<br>
                    ✅ Week columns with month names
                </div>
            </div>
        """, unsafe_allow_html=True)

    if uploaded_file and PINECONE_API_KEY:
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            process_clicked = st.button("⚡ Process & Ingest Sheets")

        if process_clicked:
            progress_bar = st.progress(0)

            st.markdown("""
                <div style="
                    background: rgba(0,166,81,0.06);
                    border: 1px solid rgba(0,166,81,0.2);
                    border-radius: 14px;
                    padding: 20px 24px;
                    margin-top: 16px;
                ">
                    <div style="color: rgba(255,255,255,0.8); font-size: 13px;
                                font-weight: 600; margin-bottom: 12px;">
                        🔄 Processing Pipeline
                    </div>
            """, unsafe_allow_html=True)

            status_text = st.empty()

            with st.spinner("Executing server-side matrix mapping..."):
                try:
                    status_text.markdown(
                        "<p style='color:rgba(255,255,255,0.5); font-size:13px;'>📖 Reading Excel sheets...</p>",
                        unsafe_allow_html=True
                    )
                    file_bytes  = BytesIO(uploaded_file.read())
                    excel_file  = pd.ExcelFile(file_bytes)
                    all_sheets  = excel_file.sheet_names

                    pc    = Pinecone(api_key=PINECONE_API_KEY)
                    index = pc.Index(PINECONE_INDEX_NAME)

                    payload_batch             = []
                    text_inputs_for_embedding = []
                    discovered_data_summary   = {}

                    progress_bar.progress(10)
                    status_text.markdown(
                        "<p style='color:rgba(255,255,255,0.5); font-size:13px;'>🗂️ Parsing sheet structure...</p>",
                        unsafe_allow_html=True
                    )

                    for sheet_name in all_sheets:
                        sheet_clean   = sheet_name.strip()
                        inferred_year = infer_year_for_sheet(sheet_clean, all_sheets)
                        if not inferred_year:
                            continue

                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

                        cat_col = find_category_column(df.columns)
                        if not cat_col:
                            continue

                        week_cols = [c for c in df.columns if 'week' in c.lower()]

                        for idx, row in df.iterrows():
                            raw_category = row.get(cat_col, None)
                            category     = normalize_category(raw_category)

                            if not category or str(raw_category).strip().lower() in EMPTY_VALUES:
                                continue

                            is_positive = category in POSITIVE_CATEGORIES
                            is_negative = category in NEGATIVE_CATEGORIES

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
                                    context_chunk = (
                                        f"Year: {inferred_year}. "
                                        f"Month: {row_month}. "
                                        f"Week: {col}. "
                                        f"Case Category: {category}. "
                                        f"Feedback: {bullet}."
                                    )
                                    metadata_payload = {
                                        "text":      context_chunk,
                                        "month":     row_month,
                                        "year":      inferred_year,
                                        "week":      col,
                                        "category":  category,
                                        "sentiment": (
                                            "positive" if is_positive
                                            else "negative" if is_negative
                                            else "neutral"
                                        ),
                                        "value": bullet
                                    }
                                    clean_cat   = re.sub(r'[^a-zA-Z0-9]', '', category.replace(' ', '_'))
                                    clean_col   = re.sub(r'[^a-zA-Z0-9]', '', col.replace(' ', '_'))
                                    clean_sheet = re.sub(r'[^a-zA-Z0-9]', '', sheet_clean.replace(' ', '_'))
                                    vector_id   = f"v_{clean_sheet}_{clean_cat}_{clean_col}_{idx}_{b_idx}"

                                    payload_batch.append({"id": vector_id, "metadata": metadata_payload})
                                    text_inputs_for_embedding.append(context_chunk)

                    total_records = len(payload_batch)
                    if total_records == 0:
                        st.warning("⚠️ No records found. Check sheet names include a year (e.g. 2025).")
                        st.stop()

                    progress_bar.progress(40)
                    status_text.markdown(
                        f"<p style='color:rgba(255,255,255,0.5); font-size:13px;'>🧠 Generating embeddings for <b style='color:#00a651'>{total_records}</b> records...</p>",
                        unsafe_allow_html=True
                    )

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
                        pct = 40 + int((i / total_records) * 35)
                        progress_bar.progress(min(pct, 75))

                    progress_bar.progress(80)
                    status_text.markdown(
                        "<p style='color:rgba(255,255,255,0.5); font-size:13px;'>📤 Upserting vectors to Pinecone...</p>",
                        unsafe_allow_html=True
                    )

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

                    progress_bar.progress(100)
                    status_text.empty()

                    # ── Success summary ──
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("""
                        <div style="
                            background: rgba(0,166,81,0.12);
                            border: 1px solid rgba(0,166,81,0.4);
                            border-radius: 14px;
                            padding: 18px 22px;
                            margin-top: 14px;
                            display: flex;
                            align-items: center;
                            gap: 14px;
                        ">
                            <div style="font-size: 2rem;">🎉</div>
                            <div>
                                <div style="color: #00d468; font-weight: 700; font-size: 15px;">
                                    Pipeline Complete!
                                </div>
                                <div style="color: rgba(255,255,255,0.55); font-size: 13px; margin-top: 2px;">
                                    Successfully ingested <b style="color:white">""" + str(total_records) + """</b> records into the vector database.
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # ── Stats row ──
                    if discovered_data_summary:
                        st.markdown("<div style='margin-top: 16px;'>", unsafe_allow_html=True)
                        top_months = sorted(
                            discovered_data_summary.items(),
                            key=lambda x: x[1], reverse=True
                        )[:4]
                        cols = st.columns(len(top_months))
                        for i, (period, count) in enumerate(top_months):
                            with cols[i]:
                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="stat-num">{count}</div>
                                        <div class="stat-label">{period}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 20px 0;'>",
                unsafe_allow_html=True)

# ==========================================
# PUBLIC CHAT INTERFACE
# ==========================================
st.markdown("""
    <div class="page-title">
        <h1>📊 Syngenta Sentiment Analyzer</h1>
        <p>Query weekly feedback, complaints, and sentiment trends from the ingested dataset</p>
    </div>
""", unsafe_allow_html=True)

# ── Render chat history with custom bubbles ──
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="bubble-user">
                <div class="msg">{message["content"]}</div>
                <div class="avatar">👤</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="bubble-bot">
                <div class="avatar">🌱</div>
                <div class="msg">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)

# ── Chat input ──
user_query = st.chat_input("Ask about weekly sentiment, complaints, or product feedback...")

if user_query and user_query.strip():

    # Show user bubble immediately
    st.markdown(f"""
        <div class="bubble-user">
            <div class="msg">{user_query}</div>
            <div class="avatar">👤</div>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # ── STRICT TOPIC GUARDRAIL ──
    allowed_keywords = [
        "sentiment", "sentiments", "feedback", "product", "syngenta", "cropwise", "app",
        "price", "unavailability", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "2024", "2025", "2026", "complaint", "complaints", "positive", "negative",
        "grower", "advisory", "quantis", "isabion", "week", "1st", "2nd", "3rd",
        "4th", "5th", "first", "second", "third", "fourth", "fifth",
        "issues", "concerns", "problems", "appreciation", "praise"
    ]
    query_words = re.findall(r'\b\w+\b', user_query.lower())
    is_relevant = any(word in allowed_keywords for word in query_words)

    if not is_relevant:
        reply = (
            "I cannot generate this response. "
            "I am strictly locked to analyzed dataset metrics "
            "and cannot find relevant information for this query."
        )
        st.markdown(f"""
            <div class="bubble-bot">
                <div class="avatar">🌱</div>
                <div class="msg">{reply}</div>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.stop()

    if not PINECONE_API_KEY:
        reply = "🤖 Execution Halted: Pinecone API key is not configured."
        st.markdown(f"""
            <div class="bubble-bot">
                <div class="avatar">🌱</div>
                <div class="msg">{reply}</div>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.spinner("Searching dataset..."):

        detected_month = None
        detected_year  = None
        detected_week  = None
        query_lower    = user_query.lower()

        for shortcut in sorted(MONTH_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(shortcut) + r'\b', query_lower):
                detected_month = MONTH_MAP[shortcut]
                break

        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        if year_match:
            detected_year = year_match.group(0)

        week_match = re.search(
            r'\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\s+week\b',
            query_lower
        )
        if week_match:
            detected_week = week_match.group(0)

        complaint_keywords = [
            "complaint", "complaints", "negative feedback",
            "negative", "issues", "problems", "concerns",
            "issue", "problem"
        ]
        positive_keywords = [
            "positive feedback", "appreciation", "praise",
            "favorable", "satisfied"
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
        elif any(word in query_lower for word in sentiment_keywords):
            query_intent = "sentiment"

        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

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

        target_year        = detected_year
        fallback_triggered = False
        latest_index_year  = None

        if detected_month and not target_year:
            latest_index_year = get_latest_year_from_index(index)
            target_year       = latest_index_year

            pos, neg, neut = query_pinecone_for_timeframe(
                index, query_vector, detected_month, target_year, detected_week, query_intent
            )

            if (len(pos) + len(neg) + len(neut)) == 0:
                try:
                    fallback_year = str(int(latest_index_year) - 1)
                    pos_fb, neg_fb, neut_fb = query_pinecone_for_timeframe(
                        index, query_vector, detected_month, fallback_year, detected_week, query_intent
                    )
                    if (len(pos_fb) + len(neg_fb) + len(neut_fb)) > 0:
                        target_year        = fallback_year
                        fallback_triggered = True
                except ValueError:
                    pass

        positive_bullets, negative_bullets, neutral_bullets = query_pinecone_for_timeframe(
            index, query_vector, detected_month, target_year, detected_week, query_intent
        )
        total_found = len(positive_bullets) + len(negative_bullets) + len(neutral_bullets)

    if fallback_triggered:
        st.info(
            f"ℹ️ No data for {detected_month} {latest_index_year}. "
            f"Falling back to **{target_year}**."
        )
    elif detected_month and not detected_year:
        st.info(
            f"ℹ️ Year not specified. Using latest available: **{target_year}**"
        )

    timeframe_label = " ".join(
        filter(None, [detected_week, detected_month, target_year])
    ) or "the requested period"

    if query_intent == "complaint":
        header = f"**Complaints of {timeframe_label}:**\n\n"
    elif query_intent == "positive":
        header = f"**Positive Feedback of {timeframe_label}:**\n\n"
    else:
        header = f"**Sentiments of {timeframe_label}:**\n\n"

    if total_found == 0:
        if detected_month or target_year or detected_week:
            reply = f"No data found for **{timeframe_label}** in the ingested dataset."
        else:
            reply = (
                "I cannot generate this response. "
                "I am strictly locked to analyzed dataset metrics "
                "and cannot find relevant information for this query."
            )
        st.markdown(f"""
            <div class="bubble-bot">
                <div class="avatar">🌱</div>
                <div class="msg">{header}{reply}</div>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": header + reply})
        st.stop()

    MAX_BULLETS = 12
    if query_intent == "complaint":
        positive_bullets = []
        negative_bullets = negative_bullets[:MAX_BULLETS]
        neutral_bullets  = neutral_bullets[:MAX_BULLETS]
    elif query_intent == "positive":
        positive_bullets = positive_bullets[:MAX_BULLETS]
        negative_bullets = []
        neutral_bullets  = []
    else:
        positive_bullets = positive_bullets[:MAX_BULLETS]
        negative_bullets = negative_bullets[:MAX_BULLETS]
        neutral_bullets  = neutral_bullets[:MAX_BULLETS]

    context_parts = []
    if positive_bullets:
        context_parts.append("POSITIVE DATA:\n" + "\n".join(positive_bullets))
    if negative_bullets:
        context_parts.append("NEGATIVE DATA:\n" + "\n".join(negative_bullets))
    if neutral_bullets:
        context_parts.append("OTHER DATA:\n"    + "\n".join(neutral_bullets))

    combined_context = "\n\n".join(context_parts)

    if query_intent == "complaint":
        system_prompt = (
            "You are a smart chatbot analyst for Syngenta. "
            "Respond naturally like a chatbot, not a formal report. "
            f"Start your response with a line like: 'The complaints for {timeframe_label} are as follows:' "
            "Cover ONLY complaints and concerns. Do NOT mention any positive feedback. "
            "Explicitly name every product and state the exact reason for each complaint. "
            "Rules:\n- No bullet points. Prose only.\n- No bracketed dates or week labels.\n"
            "- Keep it concise (4-6 sentences max).\n- Sound like a helpful chatbot, not a corporate report."
        )
    elif query_intent == "positive":
        system_prompt = (
            "You are a smart chatbot analyst for Syngenta. "
            "Respond naturally like a chatbot, not a formal report. "
            f"Start your response with a line like: 'The positive feedback for {timeframe_label} looks great!' "
            "Cover ONLY positive sentiments and appreciation. Do NOT mention any complaints. "
            "Explicitly name every product and state the exact reason users are satisfied. "
            "Rules:\n- No bullet points. Prose only.\n- No bracketed dates or week labels.\n"
            "- Keep it concise (4-6 sentences max).\n- Sound like a helpful chatbot, not a corporate report."
        )
    else:
        system_prompt = (
            "You are a smart chatbot analyst for Syngenta. "
            "Respond naturally like a chatbot, not a formal report. "
            f"Start your response with a line like: 'Here is the sentiment overview for {timeframe_label}:' "
            "Structure your response in exactly two short paragraphs:\n\n"
            "Paragraph 1 — Favorable Sentiments: Summarize positive trends. "
            "Explicitly name every product and state exactly why users are satisfied.\n\n"
            "Paragraph 2 — Complaints & Concerns: Summarize issues. "
            "Explicitly name every product and state the exact reason for each concern.\n\n"
            "Rules:\n- No bullet points. Prose only.\n- No bracketed dates or week labels.\n"
            "- Keep each paragraph concise (3-5 sentences max).\n"
            "- Sound like a helpful chatbot, not a corporate report."
        )

    user_prompt = (
        f"Timeframe: {timeframe_label}\n\n"
        f"Data Context:\n{combined_context}\n\n"
        f"User Query: {user_query}"
    )

    # ── Stream response into bot bubble ──
    bot_placeholder = st.empty()
    full_response   = ""

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        stream = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=400,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            bot_placeholder.markdown(f"""
                <div class="bubble-bot">
                    <div class="avatar">🌱</div>
                    <div class="msg">{header}{full_response}▌</div>
                </div>
            """, unsafe_allow_html=True)

        bot_placeholder.markdown(f"""
            <div class="bubble-bot">
                <div class="avatar">🌱</div>
                <div class="msg">{header}{full_response}</div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        full_response = f"Operational Processing Error: {e}"
        bot_placeholder.markdown(f"""
            <div class="bubble-bot">
                <div class="avatar">🌱</div>
                <div class="msg">{full_response}</div>
            </div>
        """, unsafe_allow_html=True)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": header + full_response
    })