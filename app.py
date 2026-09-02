"""Streamlit app — the admin surface, and a working chat fallback.

Chat lives on Vercel; this app exists because ingestion does not belong in
a request-scoped serverless function. It runs the same `vog` package, so
the two never disagree about what a question means.
"""

import hmac
import os

# pandas is not in requirements.txt — that file is the Vercel function's
# dependency set, where pandas was 63MB of a 250MB budget, and CI now
# rejects it there. It is safe to import here regardless: streamlit hard-
# depends on pandas, and this module only ever runs under streamlit.
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from vog import compose, llm, retrieval
from vog.catalog import SUGGESTED_PROMPTS_QUICK
from vog.ingest import run_ingestion
from vog.plan import MODE_REPLY, build_plan

APP_BUILD = "2026-09-02-v14 (vog package; chat primary on Vercel)"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", None)

st.set_page_config(page_title="Voice of Grower", page_icon="🌾", layout="wide")

for key, default in (("authenticated", False), ("chat_history", []),
                     ("prior_context", None), ("followups", [])):
    st.session_state.setdefault(key, default)

st.markdown(""" <style> .stApp { background: linear-gradient(180deg, #f3f9f1 0%, #eaf4e6 100%); } section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1b3a24 0%, #0f2417 100%); } section[data-testid="stSidebar"] * { color: #eef7ec !important; } section[data-testid="stSidebar"] input { color: #111 !important; } section[data-testid="stSidebar"] button { background-color: #2e7d32 !important; border: 1px solid #256029 !important; border-radius: 8px !important; } section[data-testid="stSidebar"] button, section[data-testid="stSidebar"] button p, section[data-testid="stSidebar"] button span, section[data-testid="stSidebar"] button div { color: #ffffff !important; } section[data-testid="stSidebar"] button:hover { background-color: #256029 !important; border-color: #1b3a24 !important; } section[data-testid="stSidebar"] button:hover, section[data-testid="stSidebar"] button:hover p, section[data-testid="stSidebar"] button:hover span, section[data-testid="stSidebar"] button:hover div { color: #ffffff !important; } .hero-title { font-size: 2.15rem; font-weight: 800; background: linear-gradient(90deg, #2e7d32, #558b2f, #33691e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.1rem; } .hero-subtitle { color: #4b5d4e; font-size: 0.97rem; margin-bottom: 1.2rem; } div[data-testid="stChatMessage"] { border-radius: 16px; padding: 0.7rem 1.1rem; margin-bottom: 0.6rem; box-shadow: 0 1px 5px rgba(46, 125, 50, 0.10); background: #f2f9f2; border: 1px solid #e2f0e2; } div[data-testid="stChatMessage"] ul { list-style: none; padding-left: 0.1rem; margin-top: 0.4rem; } div[data-testid="stChatMessage"] li { position: relative; padding-left: 1.5rem; margin-bottom: 0.45rem; line-height: 1.45; } div[data-testid="stChatMessage"] li::before { content: "🌱"; position: absolute; left: 0; top: 0; } .intent-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.55rem; } .badge-positive { background: #dff5df; color: #256029; } .badge-complaint { background: #fdeaea; color: #9c3b3b; } .badge-sentiment { background: #e3f1e6; color: #2e5d34; } .badge-comparison { background: #eee3f9; color: #5b3a94; } .badge-product { background: #fff3d6; color: #8a5a00; } .badge-suggestion { background: #e6e6fa; color: #4b3f8a; } .badge-ranking { background: #fde2c8; color: #8a4b00; } div[data-testid="stChatInput"] textarea { border-radius: 12px !important; } h1, .hero-title { display: flex; align-items: center; gap: 0.4rem; } </style> """, unsafe_allow_html=True)

# ─────────────────────────── sidebar / admin ─────────────────────────
with st.sidebar:
    st.header("⚙️ System Credentials")
    st.markdown("---")
    st.subheader("🔑 Admin Panel")
    if not st.session_state.authenticated:
        # Read from secrets/env with NO fallback. A hardcoded default here
        # would be a password published in a public repository.
        expected = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if not expected:
                st.error("Admin is disabled: ADMIN_PASSWORD is not configured for this deployment.")
            elif hmac.compare_digest(password, expected):
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

if st.session_state.authenticated:
    st.title("📥 Dataset Ingestion")
    uploaded = st.file_uploader("Upload Master Performance Log (.xlsx)", type=["xlsx"])
    if uploaded and PINECONE_API_KEY:
        purge_first = st.checkbox(
            "Replace all existing data (purge first)",
            help=("Records already in the index keep the tags they were written with. "
                  "After a change to how products, crops or categories are detected, a "
                  "purge is required for the corrections to take effect — re-ingesting "
                  "alone leaves the old records in place. This cannot be undone."),
        )
        if st.button("Process sheets"):
            with st.spinner("Parsing, embedding and upserting..."):
                try:
                    result = run_ingestion(uploaded.read(), PINECONE_API_KEY,
                                           purge_first=purge_first)
                    st.success(f"🎉 Ingested {result['total_records']} records.")
                    # Surface what the parser could not use. These used to be
                    # silent skips, so a workbook could half-ingest while the
                    # UI still reported success.
                    for item in result.get("skipped", []):
                        st.warning(f"Skipped — {item['sheet']} ({item['reason']}): {item['detail']}")
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# ────────────────────────────── helpers ──────────────────────────────

_BADGE_CLASS = {"🔀": "badge-comparison", "🏷️": "badge-product", "🐛": "badge-complaint",
                "🌻": "badge-positive", "💡": "badge-suggestion",
                "📊": "badge-ranking", "📈": "badge-ranking"}


def badge_html(badge_text: str) -> str:
    css = next((c for prefix, c in _BADGE_CLASS.items() if badge_text.startswith(prefix)),
               "badge-sentiment")
    return f'<span class="intent-badge {css}">{badge_text}</span>'


def render_chart(chart: dict | None):
    if not chart or not chart.get("labels"):
        return
    labels, values = chart["labels"], chart["values"]
    if chart["type"] == "line":
        # Real Timestamps, not "Month Year" strings: Vega-Lite treats a
        # string axis as categorical and sorts it alphabetically.
        index = pd.DatetimeIndex([pd.Timestamp(f"1 {l}") for l in labels], name="Month")
        st.line_chart(pd.DataFrame({"Count": values}, index=index))
    else:
        df = pd.DataFrame({"Value": values}, index=labels)
        # Same quirk for bars: hold the ranked order instead of alphabetizing.
        df.index = pd.CategoricalIndex(df.index, categories=labels, ordered=True)
        st.bar_chart(df)


def render_downloads(answer, summary_text: str, key_prefix: str):
    if not answer.export_rows:
        return
    files = compose.build_exports(answer, summary_text)
    specs = [("csv", "⬇️ CSV", "chart_data.csv", "text/csv"),
             ("excel", "⬇️ Excel", "vog_export.xlsx",
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
             ("pptx", "⬇️ PowerPoint", "vog_report.pptx",
              "application/vnd.openxmlformats-officedocument.presentationml.presentation")]
    for col, (kind, label, filename, mime) in zip(st.columns(3), specs):
        if files.get(kind):
            col.download_button(label, data=files[kind], file_name=filename,
                                mime=mime, key=f"{key_prefix}_{kind}")


def answer_for(query: str):
    """Question -> (plan, evidence, answer). Same three calls the API makes."""
    # A capability or off-topic question is answered without retrieval.
    prior = st.session_state.prior_context
    plan = build_plan(query, prior_context=prior)
    if plan.mode == MODE_REPLY:
        return plan, retrieval.Evidence(), compose.compose(plan, retrieval.Evidence())

    pc, index = retrieval.connect(PINECONE_API_KEY)
    latest = retrieval.dataset_extent(index)
    plan = build_plan(query, latest_month_year=latest, prior_context=prior)

    # Only worth a classification round trip when the regexes found nothing.
    # Whatever comes back is validated against the real catalogs first.
    if plan.needs_assist and GROQ_API_KEY:
        assist = llm.classify_query(query, GROQ_API_KEY)
        if assist:
            plan = build_plan(query, latest_month_year=latest,
                              prior_context=prior, assist=assist)

    evidence = retrieval.gather(plan, index, pc)
    return plan, evidence, compose.compose(plan, evidence)


# ──────────────────────────────── chat ───────────────────────────────
st.markdown('<div class="hero-title">🌾 Voice of Grower</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Ask about sentiment, complaints, products, crops, '
            'or trends — grounded entirely in your grower feedback data.</div>',
            unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.caption("Not sure what to ask? Try one of these:")
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS_QUICK):
        if cols[i % 2].button(prompt, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_query = prompt
            st.rerun()
else:
    if st.button("🔄 New chat"):
        st.session_state.chat_history = []
        st.session_state.prior_context = None
        st.session_state.followups = []
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        # Only the badge we construct is trusted HTML. Model output and
        # data-derived text render as plain markdown, so a script tag in an
        # ingested spreadsheet cell cannot execute here.
        if message.get("badge_html"):
            st.markdown(message["badge_html"], unsafe_allow_html=True)
        st.markdown(message["content"])

user_query = st.chat_input("Ask about sentiment, a product, or compare periods...") \
    or st.session_state.pop("pending_query", None)

if user_query and user_query.strip():
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.followups = []

    if not PINECONE_API_KEY:
        st.error("Search is not configured (PINECONE_API_KEY is unset).")
        st.stop()

    with st.spinner("Searching matching feedback records..."):
        plan, evidence, answer = answer_for(user_query)

    st.session_state.prior_context = dict(answer.context)

    if answer.kind in ("reply", "no_data"):
        with st.chat_message("assistant"):
            st.markdown(answer.text)
        st.session_state.chat_history.append({"role": "assistant", "content": answer.text})

    elif answer.kind != "prompt":
        # Ranking and trend: the numbers are already final. The model may
        # only add colour on top of them.
        badge = badge_html(answer.badge)
        text = answer.text
        if answer.top and GROQ_API_KEY:
            extra = compose.narrate_result(
                plan.rank_dimension or "month", answer.top[0], answer.top[1],
                [str((m.get("metadata") or {}).get("value", "")) for m in evidence.matches[:200]],
                GROQ_API_KEY)
            if extra:
                text += f"\n\n{extra}"
        with st.chat_message("assistant"):
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(text)
            render_chart(answer.chart)
            render_downloads(answer, text, key_prefix=answer.kind)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": text, "badge_html": badge})

    else:
        badge = badge_html(answer.badge)
        with st.chat_message("assistant"):
            st.markdown(badge, unsafe_allow_html=True)
            box = st.empty()
            body = ""
            failure = ""
            try:
                for chunk in llm.stream_answer(answer.system_prompt, answer.user_prompt,
                                               GROQ_API_KEY, max_tokens=answer.token_budget):
                    body += (chunk.choices[0].delta.content or "") if chunk.choices else ""
                    box.markdown(answer.header + body + "▌")
                box.markdown(answer.header + body)
            except Exception:
                body, failure = "", ("The answer could not be generated — the language "
                                     "model is unavailable right now. Please try again "
                                     "in a moment.")
            if not body.strip():
                # An empty completion looks exactly like a working app with
                # nothing to say, so name it rather than rendering a blank.
                failure = failure or ("The model returned an empty response. "
                                      "Please try again.")
                box.markdown(failure)

            if failure:
                # Don't build a deck out of an error message, and don't spend
                # a second model call suggesting follow-ups to a failure.
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": failure})
            else:
                full = answer.header + body
                render_chart(answer.chart)
                render_downloads(answer, body, key_prefix="qa")
                st.session_state.prior_context = {**answer.context, "last_reply": full[:1500]}
                st.session_state.followups = compose.followups(plan, answer, full, GROQ_API_KEY)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full, "badge_html": badge})

if st.session_state.followups:
    st.caption("Ask next:")
    for i, suggestion in enumerate(st.session_state.followups):
        if st.button(suggestion, key=f"followup_{i}", use_container_width=True):
            st.session_state.pending_query = suggestion
            st.rerun()
