"""The pre-rewrite public surface, re-expressed over the `vog` package.

This exists so the regression suite written against the old monolith keeps
running unchanged against the new code — the assertions in it are the
record of every bug that was found and fixed, and rewriting 200+ of them
by hand would have been the most likely way to lose one.

It renames and re-shapes; it contains no logic of its own. A behaviour
change therefore shows up as a failing assertion rather than being
absorbed here. New tests should import `vog.*` directly.
"""

from vog import compose, exports, llm, parsing, retrieval
from vog import catalog
from vog.catalog import *          # noqa: F401,F403  — constants by old name
from vog.ingest import run_ingestion  # noqa: F401
from vog.parsing import *          # noqa: F401,F403
from vog.parsing import (          # underscore names `import *` skips
    _blend_vectors, _clean_cell_text, _explicit_list_format, _idx_to_month_year,
    _month_idx, _interleave_by_recency, _wants_products_only, _week_number,
)  # noqa: F401
from vog.plan import _rank_top_n, _resolve_window, build_plan

from conftest import run_query as process_chat_query  # noqa: F401

# Renamed in the rewrite.
AGGREGATION_PAGE_SIZE = retrieval.AGGREGATION_TOP_K
build_csv_bytes = exports.build_csv_bytes
build_excel_bytes = exports.build_excel_bytes


def detect_requested_top_n(query_lower: str) -> int | None:
    """Old contract returned None when no N was stated; the planner now
    defaults to 10, so ask it for a sentinel and translate."""
    n = _rank_top_n(query_lower, default=-1)
    return None if n == -1 else n


def build_pptx_report(title, subtitle, exec_summary_lines, kpis, chart_title,
                      chart_labels, chart_values, chart_type="column",
                      table_headers=None, table_rows=None,
                      insights=None, recommendations=None):
    # insights/recommendations were folded into summary_lines.
    lines = list(exec_summary_lines or [])
    for extra in (insights, recommendations):
        if extra:
            lines += list(extra)
    return exports.build_pptx_bytes(
        title=title, subtitle=subtitle, summary_lines=lines, kpis=kpis,
        chart_title=chart_title, chart_labels=chart_labels,
        chart_values=chart_values, chart_type=chart_type,
        table_headers=table_headers, table_rows=table_rows)


def fetch_matches_for_aggregation(index, filter_conditions, top_k=None):
    """(matches, complete). `complete` is False when the fetch hit its
    ceiling, so any count derived from it is a lower bound."""
    limit = top_k or retrieval.AGGREGATION_TOP_K
    res = index.query(vector=[0.0] * catalog.EMBEDDING_DIMENSION, top_k=limit,
                      include_metadata=True, filter=filter_conditions or None)
    raw = res.get("matches", [])
    matches = [m for m in raw if not (m.get("metadata") or {}).get("is_stats_record")]
    return matches, len(raw) < limit


def resolve_relative_window(window: dict, index):
    """Old signature took the index and looked the newest month up itself."""
    latest = retrieval.dataset_extent(index)
    if not latest:
        return None
    timeframe = _resolve_window(window, latest)
    return (timeframe.months, timeframe.label) if timeframe else None


def llm_assisted_query_understanding(user_query, groq_api_key):
    return llm.classify_query(user_query, groq_api_key) if groq_api_key else None


def generate_deterministic_narrative(dimension_label, top_name, top_count,
                                     bullets, groq_api_key):
    if not groq_api_key:
        return ""
    return compose.narrate_result(dimension_label, top_name, top_count,
                                  bullets, groq_api_key)


def generate_followup_suggestions(query_intent, subject_label, timeframe_label,
                                  full_response, groq_api_key, max_suggestions=3):
    if not groq_api_key:
        return []
    plan = build_plan(f"{query_intent} feedback")
    plan.intent = query_intent
    return compose.suggest_followups(plan, full_response, groq_api_key,
                                     n=max_suggestions)


def finalize_normal_response(state: dict, full_response: str) -> dict:
    """Exports are built on demand now instead of eagerly on every turn, so
    this is the same work, just deferred to the point of use."""
    answer = state["_answer"]
    return {
        "final_reply": (answer.header or "") + full_response,
        "chart": answer.chart,
        "downloads": compose.build_exports(answer, full_response),
        "kpis": answer.export_meta.get("kpis", {}),
        "export_rows": answer.export_rows,
    }
