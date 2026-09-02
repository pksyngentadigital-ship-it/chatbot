#!/usr/bin/env python3
"""
Stage 0 go/no-go: can the configured model reliably call tools?

The whole agentic rebuild rests on one assumption — that the model can
pick the right tool and fill in the right arguments. This answers that in
a few minutes, with no integration and nothing touching production
behaviour, so we find out before committing days to it.

Deliberately NOT a test of answer quality. It only asks: given a real
question and real tool schemas, does the model choose correctly?

Run locally:
    GROQ_API_KEY=... python scripts/toolcall_probe.py

Or hit the admin endpoint (uses the server's configured key):
    GET /admin/toolcall-probe
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vog.catalog import GROQ_MODEL  # noqa: E402


# ── Tool schemas ──
# These mirror the six tools in the scope doc, trimmed to the three that
# exercise the interesting decisions: retrieval vs counting vs time series.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_feedback",
            "description": (
                "Retrieve individual grower feedback comments. Use for open questions "
                "about what people said. Does NOT count or rank."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["complaint", "positive", "suggestion", "sentiment", "topics"],
                        "description": "Which kind of feedback to retrieve.",
                    },
                    "product": {"type": "string", "description": "Exact product name, if the question names one."},
                    "crop": {"type": "string", "description": "Exact crop name, if the question names one."},
                    "month": {"type": "string", "description": "Full month name, e.g. January."},
                    "year": {"type": "string", "description": "Four-digit year."},
                },
                "required": ["intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_by",
            "description": (
                "Count and rank crops or products by how often they appear. Use whenever "
                "the question asks which/most/top/highest/frequency. Returns exact counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": ["crop", "product"]},
                    "intent": {
                        "type": "string",
                        "enum": ["complaint", "positive", "suggestion", "sentiment"],
                    },
                    "top_n": {"type": "integer", "description": "How many rows to return."},
                    "month": {"type": "string"},
                    "year": {"type": "string"},
                },
                "required": ["dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_trend",
            "description": (
                "Month-by-month counts over time, with month-over-month change. Use for "
                "trend/over-time/growth questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["complaint", "positive", "suggestion", "sentiment"],
                    },
                    "product": {"type": "string"},
                    "crop": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are the query planner for a grower-feedback analytics assistant. "
    "Decide which tool(s) to call to answer the user's question. "
    "Use rank_by for any 'which/most/top/highest' question — never try to count "
    "by reading individual comments. Use get_monthly_trend for anything about "
    "change over time. Use search_feedback for open questions about what people said. "
    "If a question needs several things, call several tools. "
    "Only use product or crop names exactly as the user wrote them."
)

# (question, expected_tool, {required_arg: expected_value or None for "any"} , min_calls)
CASES = [
    ("What are growers saying about Isabion?",
     "search_feedback", {"product": "isabion"}, 1),
    ("Which crop generated the highest number of complaints?",
     "rank_by", {"dimension": "crop", "intent": "complaint"}, 1),
    ("Show the monthly complaint trend",
     "get_monthly_trend", {"intent": "complaint"}, 1),
    ("What are the top 5 products by complaint frequency?",
     "rank_by", {"dimension": "product", "intent": "complaint", "top_n": 5}, 1),
    ("What do growers like about Axial?",
     "search_feedback", {"product": "axial", "intent": "positive"}, 1),
    ("What complaints came from wheat growers in January 2026?",
     "search_feedback", {"crop": "wheat", "intent": "complaint", "month": "January", "year": "2026"}, 1),
    ("What suggestions have growers made?",
     "search_feedback", {"intent": "suggestion"}, 1),
    ("How has positive feedback for Cropwise changed over time?",
     "get_monthly_trend", {"product": "cropwise", "intent": "positive"}, 1),
    # Multi-step: the real test of composition.
    ("Compare customer sentiment for Tilt and Isabion",
     "search_feedback", {}, 2),
    ("What are the top 3 products by complaints, and how did each trend over time?",
     "rank_by", {"dimension": "product"}, 2),
]


def _norm(v):
    return str(v).strip().lower() if v is not None else None


def evaluate(calls, expected_tool, expected_args, min_calls):
    """Score one case. Returns (passed, detail)."""
    if not calls:
        return False, "no tool call at all"

    names = [c["name"] for c in calls]
    if expected_tool not in names:
        return False, f"expected {expected_tool}, got {names}"

    if len(calls) < min_calls:
        return False, f"expected >={min_calls} calls, got {len(calls)} ({names})"

    # Check args on the first matching call.
    call = next(c for c in calls if c["name"] == expected_tool)
    args = call.get("args") or {}
    missing = []
    for key, want in expected_args.items():
        got = args.get(key)
        if got is None:
            missing.append(f"{key} missing")
        elif want is not None and _norm(got) != _norm(want):
            missing.append(f"{key}={got!r} (wanted {want!r})")
    if missing:
        return False, "; ".join(missing)
    return True, f"{len(calls)} call(s): {names}"


def run_probe(groq_api_key: str, model: str | None = None) -> dict:
    from groq import Groq

    model = model or GROQ_MODEL
    client = Groq(api_key=groq_api_key)
    results, passed = [], 0

    for question, exp_tool, exp_args, min_calls in CASES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=600,
            )
            msg = resp.choices[0].message
            calls = []
            for tc in (getattr(msg, "tool_calls", None) or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except ValueError:
                    args = {}
                calls.append({"name": tc.function.name, "args": args})
            ok, detail = evaluate(calls, exp_tool, exp_args, min_calls)
        except Exception as e:
            ok, detail, calls = False, f"ERROR: {type(e).__name__}: {e}", []

        passed += 1 if ok else 0
        results.append({
            "question": question,
            "expected_tool": exp_tool,
            "passed": ok,
            "detail": detail,
            "calls": calls,
        })

    total = len(CASES)
    rate = round(100 * passed / total)
    if rate >= 90:
        verdict = "GO — tool calling is reliable enough to build on."
    elif rate >= 70:
        verdict = "MARGINAL — workable but expect to constrain the loop tightly and re-test."
    else:
        verdict = "NO-GO — this model cannot plan reliably; do not build the agent on it."

    return {
        "model": model,
        "passed": passed,
        "total": total,
        "pass_rate_pct": rate,
        "verdict": verdict,
        "results": results,
    }


def main():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY is not set.\n"
              "Either export it, or call the admin endpoint /admin/toolcall-probe "
              "which uses the server's configured key.", file=sys.stderr)
        sys.exit(2)

    report = run_probe(key, os.getenv("GROQ_MODEL"))
    print(f"\nModel: {report['model']}")
    print(f"Passed {report['passed']}/{report['total']}  ({report['pass_rate_pct']}%)\n")
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['question']}")
        print(f"         {r['detail']}")
    print(f"\n{report['verdict']}\n")
    sys.exit(0 if report["pass_rate_pct"] >= 70 else 1)


if __name__ == "__main__":
    main()
