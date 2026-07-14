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
RELEVANCE_THRESHOLD = 0.30  # ← tune this if needed

st.set_page_config(page_title="Weekly Sentiment RAG Engine", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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


# ==========================================
# SEMANTIC RELEVANCE GUARDRAIL
# Replaces keyword-based filtering entirely.
# Uses Pinecone vector similarity to decide
# if the query is relevant to the dataset.
# ==========================================
def is_query_relevant(pc, query_vector: list, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """
    Embeds the user query and checks if Pinecone
    returns at least one match above the threshold.
    If yes → relevant to dataset → proceed.
    If no  → off-topic → block with guardrail message.
    Works for any language, any product name, any phrasing.
    """
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=False
        )
        matches = results.get("matches", [])
        if not matches:
            return False
        top_score = matches[0].get("score", 0.0)
        return top_score >= threshold
    except Exception:
        # If check fails, allow query to proceed
        # to avoid blocking valid queries on API errors
        return True


# ==========================================
# FIXED: query_intent parameter added
# ==========================================
def query_pinecone_for_timeframe(index, query_vector, month, year, week, query_intent="sentiment"):
    filter_conditions = {}
    if month:
        filter_conditions["month"] = {"$eq": month}
    if year:
        filter_conditions["year"]  = {"$eq": year}

    # ── SENTIMENT FILTER AT DATABASE LEVEL ──
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
st.title("📊 Strategic Enterprise Performance Analyzer")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Query specific week, month, or portfolio sentiment tracking metrics...")

if user_query and user_query.strip():
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if not PINECONE_API_KEY:
        with st.chat_message("assistant"):
            st.markdown("🤖 Execution Halted: Pinecone API key is not configured.")
        st.stop()

    # ==========================================
    # STEP 1: EMBED QUERY FIRST (used for both
    # guardrail check AND retrieval below)
    # ==========================================
    with st.spinner("Searching and aggregating matching historical data records..."):

        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            query_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[user_query],
                parameters={"input_type": "query", "dimension": EMBEDDING_DIMENSION}
            )
            query_vector = query_response[0].values
        except Exception as e:
            st.error(f"Query embedding failed: {e}")
            st.stop()

        # ==========================================
        # STEP 2: SEMANTIC GUARDRAIL
        # No keyword list. Pure vector similarity.
        # Handles any language, any product name,
        # any phrasing automatically.
        # ==========================================
        if not is_query_relevant(pc, query_vector, threshold=RELEVANCE_THRESHOLD):
            reply = (
                "I cannot generate this response. "
                "I am strictly locked to analyzed dataset metrics "
                "and cannot find relevant information for this query."
            )
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.stop()

        # ==========================================
        # STEP 3: TIMEFRAME & INTENT DETECTION
        # (unchanged from original)
        # ==========================================
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
            "complaint", "complaints", "negative", "issues",
            "problems", "concerns", "issue", "problem"
        ]
        positive_keywords = [
            "positive", "appreciation", "praise",
            "favorable", "good feedback", "satisfied"
        ]

        query_intent = "sentiment"
        if any(word in query_lower for word in complaint_keywords):
            query_intent = "complaint"
        elif any(word in query_lower for word in positive_keywords):
            query_intent = "positive"

        index = pc.Index(PINECONE_INDEX_NAME)

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
            f"ℹ️ No data found for {detected_month} {latest_index_year}. "
            f"Automatically falling back to **{target_year}**."
        )
    elif detected_month and not detected_year:
        st.info(
            f"ℹ️ Year not specified. Defaulting to the latest available dataset year: **{target_year}**"
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
            reply = (
                f"{header}"
                f"No data found for **{timeframe_label}** in the ingested dataset."
            )
        else:
            reply = (
                "I cannot generate this response. "
                "I am strictly locked to analyzed dataset metrics "
                "and cannot find relevant information for this query."
            )
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
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
            "You are an expert agricultural portfolio analyst for Syngenta. "
            "The user is asking specifically about complaints and negative feedback. "
            "Write a professional, natural response covering ONLY complaints and concerns. "
            "Do NOT mention any positive feedback whatsoever.\n\n"
            "Write one focused paragraph:\n"
            "Explicitly name every product mentioned in the data and state the exact reason "
            "for each complaint (e.g., price issues, unavailability, poor results, zero efficacy).\n\n"
            "Rules:\n"
            "- No bullet points. Prose only.\n"
            "- No bracketed dates or week labels.\n"
            "- Keep it concise (4-6 sentences max).\n"
            "- Every product in the context must appear with its specific complaint reason."
        )
    elif query_intent == "positive":
        system_prompt = (
            "You are an expert agricultural portfolio analyst for Syngenta. "
            "The user is asking specifically about positive feedback and appreciation. "
            "Write a professional, natural response covering ONLY positive sentiments. "
            "Do NOT mention any complaints or negative feedback whatsoever.\n\n"
            "Write one focused paragraph:\n"
            "Explicitly name every product mentioned in the data and state the exact reason "
            "why farmers or users are satisfied with it.\n\n"
            "Rules:\n"
            "- No bullet points. Prose only.\n"
            "- No bracketed dates or week labels.\n"
            "- Keep it concise (4-6 sentences max).\n"
            "- Every product in the context must appear with its specific positive reason."
        )
    else:
        system_prompt = (
            "You are an expert agricultural portfolio analyst for Syngenta. "
            "Analyze the provided feedback data and write a professional, natural chatbot response. "
            "Do NOT include raw metadata tags, bracketed weeks, or IDs in your output. "
            "Write in clean, flowing English sentences only.\n\n"
            "Structure your response in exactly two short paragraphs:\n\n"
            "Paragraph 1 — Favorable Sentiments: Summarize positive trends. "
            "Explicitly name every product in the positive data and state exactly why users are satisfied.\n\n"
            "Paragraph 2 — Complaints & Concerns: Summarize issues and queries. "
            "Explicitly name every product in the negative/query data and state the exact reason for each concern.\n\n"
            "Rules:\n"
            "- No bullet points. Prose only.\n"
            "- No bracketed dates or week labels.\n"
            "- Keep each paragraph concise (3-5 sentences max).\n"
            "- Every product in the context must appear in your response with its reason."
        )

    user_prompt = (
        f"Timeframe: {timeframe_label}\n\n"
        f"Data Context:\n{combined_context}\n\n"
        f"User Query: {user_query}"
    )

    # ── Stream response with Groq ──
    with st.chat_message("assistant"):
        stream_box    = st.empty()
        full_response = ""

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
                stream_box.markdown(header + full_response + "▌")

            stream_box.markdown(header + full_response)

        except Exception as e:
            full_response = f"Operational Processing Error: {e}"
            stream_box.markdown(header + full_response)

    final_reply = header + full_response
    st.session_state.chat_history.append({"role": "assistant", "content": final_reply})