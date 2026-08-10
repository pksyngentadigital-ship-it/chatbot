import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

import vog_core

# ── APP BUILD MARKER ── (bump this string whenever the file is regenerated,
# so it's easy to confirm in the sidebar/logs which version is deployed)
APP_BUILD = "2026-07-15-v12 (refactored onto vog_core.py — shared logic for a future non-Streamlit UI)"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", None)

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
# ADMIN: EXCEL INGESTION
# ==========================================
if st.session_state.authenticated:
    st.title("📥 Dataset Pipeline Ingestion Panel")
    uploaded_file = st.file_uploader("Upload Master Performance Log (.xlsx)", type=["xlsx"])

    if uploaded_file and PINECONE_API_KEY:
        if st.button("Process Sheets & Map Matrix"):
            with st.spinner("Executing server-side matrix mapping..."):
                try:
                    result = vog_core.run_ingestion(uploaded_file.read(), PINECONE_API_KEY)
                    st.success(f"🎉 Pipeline complete! Ingested {result['total_records']} records.")
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Inbound data ingestion pipe error: {e}")

# ==========================================
# UI HELPERS
# ==========================================

def badge_html(badge_text: str) -> str:
    """Map a plain-text badge (e.g. '🐛 Complaints') to a styled pill matching its intent."""
    css_class = "badge-sentiment"
    if badge_text.startswith("🔀"):
        css_class = "badge-comparison"
    elif badge_text.startswith("🏷️"):
        css_class = "badge-product"
    elif badge_text.startswith("🐛"):
        css_class = "badge-complaint"
    elif badge_text.startswith("🌻"):
        css_class = "badge-positive"
    elif badge_text.startswith("💡"):
        css_class = "badge-suggestion"
    elif badge_text.startswith("📊") or badge_text.startswith("📈"):
        css_class = "badge-ranking"
    return f'<span class="intent-badge {css_class}">{badge_text}</span>'


def render_chart(chart_meta: dict | None):
    if not chart_meta or not chart_meta.get("labels"):
        return
    labels = chart_meta["labels"]
    values = chart_meta["values"]
    if chart_meta["type"] == "line":
        # Real Timestamps (not the "Month Year" strings) so the chart's x-axis
        # sorts chronologically — Vega-Lite (which st.line_chart renders
        # through) treats a plain string axis as categorical and sorts it
        # alphabetically by default.
        month_dates = [pd.Timestamp(f"1 {l}") for l in labels]
        df = pd.DataFrame({"Count": values}, index=pd.DatetimeIndex(month_dates, name="Month"))
        st.line_chart(df)
    else:
        df = pd.DataFrame({"Value": values}, index=labels)
        # Same Vega-Lite quirk for bar charts: force the given order (ranked
        # highest-to-lowest, or the fixed Positive/Negative/Other order)
        # instead of letting it alphabetize the categories.
        df.index = pd.CategoricalIndex(df.index, categories=labels, ordered=True)
        st.bar_chart(df)


def render_downloads(downloads: dict | None, key_prefix: str):
    if not downloads:
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        if downloads.get("csv"):
            st.download_button("⬇️ Chart data (CSV)", data=downloads["csv"], file_name="chart_data.csv",
                                mime="text/csv", key=f"{key_prefix}_csv")
    with col2:
        if downloads.get("excel"):
            st.download_button("⬇️ Excel", data=downloads["excel"], file_name="vog_export.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"{key_prefix}_excel")
    with col3:
        if downloads.get("pptx"):
            st.download_button("⬇️ PowerPoint", data=downloads["pptx"], file_name="vog_report.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                key=f"{key_prefix}_pptx")


# ==========================================
# PUBLIC CHAT INTERFACE
# ==========================================
st.markdown('<div class="hero-title">🌾 Strategic Enterprise Performance Analyzer 🌱</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask about sentiment 🌾, complaints 🐛, positive feedback 🌻, a specific '
    'product 🏷️, or compare weeks / months / years 🔀.</div>',
    unsafe_allow_html=True
)

with st.expander("💡 Not sure what to ask? Try one of these", expanded=(len(st.session_state.chat_history) == 0)):
    tabs = st.tabs(list(vog_core.SUGGESTED_PROMPTS.keys()))
    for tab, (category, prompts) in zip(tabs, vog_core.SUGGESTED_PROMPTS.items()):
        with tab:
            for i, prompt in enumerate(prompts):
                if st.button(prompt, key=f"suggest_{category}_{i}", use_container_width=True):
                    st.session_state.pending_query = prompt
                    st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

typed_query = st.chat_input("Ask about sentiment, a product, or compare periods...")
user_query = typed_query or st.session_state.pop("pending_query", None)

if user_query and user_query.strip():
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Searching and aggregating matching historical data records..."):
        state = vog_core.process_chat_query(user_query, PINECONE_API_KEY, GROQ_API_KEY)

    kind = state["kind"]

    if kind in ("blocked", "no_key", "no_data"):
        reply = state["reply"]
        with st.chat_message("assistant"):
            st.markdown(reply, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    elif kind in ("ranking", "trend"):
        badge = badge_html(state["badge"])
        reply = f"{badge}\n\n{state['reply']}"
        with st.chat_message("assistant"):
            st.markdown(reply, unsafe_allow_html=True)
            render_chart(state.get("chart"))
            render_downloads(state.get("downloads"), key_prefix=kind)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    else:  # kind == "normal" — stream the LLM response ourselves
        badge = badge_html(state["badge"])
        header = state["header"]
        with st.chat_message("assistant"):
            st.markdown(badge, unsafe_allow_html=True)
            stream_box = st.empty()
            full_response = ""
            try:
                stream = vog_core.call_groq(
                    state["system_prompt"], state["user_prompt"], GROQ_API_KEY,
                    max_tokens=state["response_token_budget"]
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    full_response += token
                    stream_box.markdown(header + full_response + "▌")
                stream_box.markdown(header + full_response)
            except Exception as e:
                full_response = f"Operational Processing Error: {e}"
                stream_box.markdown(header + full_response)

            result = vog_core.finalize_normal_response(state, full_response)
            render_chart(result["chart"])
            render_downloads(result["downloads"], key_prefix="qa")

        st.session_state.chat_history.append({"role": "assistant", "content": result["final_reply"]})
