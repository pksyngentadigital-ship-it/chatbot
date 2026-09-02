"""Evidence -> a finished reply, or the prompt that will produce one.

The split that matters: anything with a checkable right answer (counts,
rankings, trends) is computed here in Python and handed to the model only
as narration material. The model never produces a number.
"""

from dataclasses import dataclass, field

from vog import exports
from vog import parsing as P
from vog.catalog import DEFAULT_TIMEFRAME_LABEL
from vog.plan import MODE_RANK, MODE_REPLY, MODE_TREND, QueryPlan
from vog.retrieval import Evidence


@dataclass
class Answer:
    kind: str                     # reply | ranking | trend | prompt | no_data
    badge: str | None = None
    header: str = ""
    text: str = ""                # finished reply, when there is one
    system_prompt: str | None = None
    user_prompt: str | None = None
    token_budget: int = 600
    chart: dict | None = None
    export_rows: list[dict] = field(default_factory=list)
    export_meta: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)


def compose(plan: QueryPlan, evidence: Evidence) -> Answer:
    if plan.mode == MODE_REPLY:
        return Answer(kind="reply", text=plan.canned_reply or "")
    if plan.mode == MODE_RANK:
        return _rank(plan, evidence)
    if plan.mode == MODE_TREND:
        return _trend(plan, evidence)
    return _answer(plan, evidence)


def _context_of(plan: QueryPlan) -> dict:
    seg = plan.segments[0] if plan.segments else None
    return {
        "product": seg.product if seg else None,
        "crop": seg.crop if seg else None,
        "intent": plan.intent,
    }


# ─────────────────────────── deterministic ───────────────────────────

def _rank(plan: QueryPlan, ev: Evidence) -> Answer:
    field_name = "crop" if plan.rank_dimension == "crop" else "products"
    ranking = P.rank_by_field(ev.matches, field_name, top_n=plan.rank_top_n)
    scope = plan.segments[0].timeframe.label if plan.segments else DEFAULT_TIMEFRAME_LABEL
    badge = f"📊 {plan.rank_dimension.title()} Ranking"
    header = f"📊 {plan.rank_dimension.title()}-wise Ranking ({scope}):\n\n"

    if not ranking:
        return Answer(
            kind="no_data", badge=badge, header=header, context=_context_of(plan),
            text=(f"{header}No {plan.rank_dimension} tags were found in the matched "
                  f"records for {scope} — nothing to rank."),
        )

    tagged = sum(c for _, c in P.rank_by_field(ev.matches, field_name, top_n=10_000))
    top_name, top_count = ranking[0]
    caveat = "" if ev.complete else (
        "\n\n*Note: the result set hit the query limit, so these counts are "
        "lower bounds rather than a complete census.*"
    )
    table = ["| Rank | " + plan.rank_dimension.title() + " | Mentions |", "|---|---|---|"]
    table += [f"| {i} | {n} | {c} |" for i, (n, c) in enumerate(ranking, start=1)]

    text = (
        f"{header}Based on {len(ev.matches)} matched records "
        f"({tagged} tagged mentions), **{top_name}** ranks highest with "
        f"{top_count} mention{'s' if top_count != 1 else ''}.\n\n"
        + "\n".join(table) + caveat
    )
    rows = [{plan.rank_dimension.title(): n, "Mentions": c} for n, c in ranking]
    return Answer(
        kind="ranking", badge=badge, header=header, text=text,
        chart={"type": "bar", "title": f"{plan.rank_dimension.title()} Mentions",
               "labels": [n for n, _ in ranking], "values": [c for _, c in ranking]}
        if plan.output_format == "chart" else None,
        export_rows=rows,
        export_meta={"title": f"{plan.rank_dimension.title()}-wise Ranking", "subtitle": scope,
                     "summary": [f"{top_name} ranks highest with {top_count} mentions."],
                     "kpis": {"Matched records": len(ev.matches), "Tagged mentions": tagged},
                     "chart_labels": [n for n, _ in ranking],
                     "chart_values": [c for _, c in ranking],
                     "chart_title": "Mentions", "chart_type": "column",
                     "table_headers": ["Rank", plan.rank_dimension.title(), "Mentions"],
                     "table_rows": [(i, n, c) for i, (n, c) in enumerate(ranking, start=1)]},
        context=_context_of(plan),
    )


def _trend(plan: QueryPlan, ev: Evidence) -> Answer:
    monthly = P.compute_monthly_trend(ev.matches)
    seg = plan.segments[0] if plan.segments else None
    subject = P.build_subject_label(seg.product if seg else None, seg.crop if seg else None)
    subject_bit = f" for {subject}" if subject else ""
    header = f"📈 Monthly Trend Analysis{subject_bit}:\n\n"

    if not monthly:
        return Answer(kind="no_data", badge="📈 Monthly Trend", header=header,
                      context=_context_of(plan),
                      text=f"{header}No dated records were found{subject_bit} to build a trend from.")

    growth = P.compute_growth_series(monthly)
    high, low = max(monthly, key=lambda x: x[1]), min(monthly, key=lambda x: x[1])
    table = ["| Month | Count | MoM Growth |", "|---|---|---|"]
    for (label, count), g in zip(monthly, growth):
        table.append(f"| {label} | {count} | {'—' if g is None else f'{g:+}%'} |")

    text = (f"{header}Highest month: **{high[0]}** ({high[1]} records). "
            f"Lowest month: **{low[0]}** ({low[1]} records).\n\n" + "\n".join(table))
    rows = [{"Month": l, "Count": c, "MoM Growth %": g} for (l, c), g in zip(monthly, growth)]
    return Answer(
        kind="trend", badge="📈 Monthly Trend", header=header, text=text,
        chart={"type": "line", "title": "Monthly Trend",
               "labels": [l for l, _ in monthly], "values": [c for _, c in monthly]}
        if plan.output_format == "chart" else None,
        export_rows=rows,
        export_meta={"title": f"Monthly Trend{subject_bit}",
                     "subtitle": f"{monthly[0][0]} – {monthly[-1][0]}",
                     "summary": [f"Highest: {high[0]} ({high[1]}).", f"Lowest: {low[0]} ({low[1]})."],
                     "kpis": {"Total records": sum(c for _, c in monthly)},
                     "chart_labels": [l for l, _ in monthly],
                     "chart_values": [c for _, c in monthly],
                     "chart_title": "Monthly Trend", "chart_type": "line",
                     "table_headers": ["Month", "Count", "MoM Growth %"],
                     "table_rows": [(l, c, "—" if g is None else f"{g}%")
                                    for (l, c), g in zip(monthly, growth)]},
        context=_context_of(plan),
    )


# ──────────────────────────── model answer ───────────────────────────

def _answer(plan: QueryPlan, ev: Evidence) -> Answer:
    seg = plan.segments[0] if plan.segments else None
    subject = plan.subject_label
    timeframe = " vs ".join(s.label for s in plan.segments) if plan.is_comparison else (
        seg.timeframe.label if seg else DEFAULT_TIMEFRAME_LABEL
    )
    # The prompt builders take periods as tuples whose [0] is the label.
    periods = [(s.label,) for s in plan.segments] if plan.is_comparison else []
    header = P.build_header(plan.intent, timeframe,
                            None if plan.is_comparison else (seg.product if seg else None),
                            periods,
                            None if plan.is_comparison else (seg.crop if seg else None),
                            category_filter=plan.category_filter)
    badge = P.build_intent_badge(plan.intent,
                                 None if plan.is_comparison else (seg.product if seg else None),
                                 periods,
                                 None if plan.is_comparison else (seg.crop if seg else None),
                                 category_filter=plan.category_filter)

    if ev.shown == 0:
        subject_bit = f" for '{subject}'" if subject else ""
        where = f" in {timeframe}" if timeframe != DEFAULT_TIMEFRAME_LABEL else ""
        return Answer(kind="no_data", badge=badge, header=header, context=_context_of(plan),
                      text=f"{header}No data found{subject_bit}{where}.")

    blocks, rows = [], []
    for s in ev.segments:
        lines = [f"=== {s.label} ==="] if plan.is_comparison else []
        for name, bucket in (("POSITIVE", s.positive), ("NEGATIVE", s.negative), ("OTHER", s.neutral)):
            if bucket:
                lines.append(f"{name} DATA:\n" + "\n".join(bucket))
        blocks.append("\n".join(lines))
        for label, bucket in (("Positive", s.positive), ("Negative", s.negative), ("Other", s.neutral)):
            rows += [{"Segment": s.label, "Sentiment": label, "Feedback": b} for b in bucket]

    # Tell the model plainly when it is seeing a sample. Saying "N total"
    # for a truncated set invited it to describe 12 of 240 as the whole
    # picture.
    sampled = ev.total > ev.shown
    scope_note = (f"{ev.shown} of {ev.total} matching points (a sample — do not "
                  f"describe these as the complete set)"
                  if sampled else f"{ev.shown} distinct data points (all of them)")

    system_prompt = P.build_system_prompt(
        plan.intent, timeframe, plan.list_format,
        None if plan.is_comparison else (seg.product if seg else None),
        periods,
        active_crop=None if plan.is_comparison else (seg.crop if seg else None),
        output_format=plan.output_format,
        wants_products_only=plan.products_only,
        avoid_repeat_text=plan.avoid_repeat,
        comparison_axis=plan.comparison_axis or "time",
    )
    user_prompt = (f"Timeframe: {timeframe}\n\n"
                   f"Data Context ({scope_note}):\n" + "\n\n".join(blocks) +
                   f"\n\nUser Query: {plan.query}")

    return Answer(
        kind="prompt", badge=badge, header=header,
        system_prompt=system_prompt, user_prompt=user_prompt,
        token_budget=900 if (plan.output_format in ("exec_summary", "table", "ppt")
                             or plan.intent == "topics") else 600,
        chart=_sentiment_chart(ev, plan) if plan.output_format == "chart" else None,
        export_rows=rows,
        export_meta={"title": subject or f"{plan.intent.title()} Analysis",
                     "subtitle": timeframe,
                     # True totals, not the truncated ones. A month with 60
                     # negatives exported "Negative: 12" before.
                     "kpis": {"Matching points": ev.total,
                              "Positive": sum(s.totals["positive"] for s in ev.segments),
                              "Negative": sum(s.totals["negative"] for s in ev.segments),
                              "Other": sum(s.totals["neutral"] for s in ev.segments)},
                     "table_headers": ["Segment", "Sentiment", "Feedback"],
                     "table_rows": [(r["Segment"], r["Sentiment"], r["Feedback"]) for r in rows]},
        context=_context_of(plan),
    )


def _sentiment_chart(ev: Evidence, plan: QueryPlan) -> dict:
    if plan.is_comparison:
        return {"type": "bar", "title": "Matching points by segment",
                "labels": [s.label for s in ev.segments],
                "values": [s.total for s in ev.segments]}
    s = ev.segments[0]
    return {"type": "bar", "title": "Sentiment breakdown",
            "labels": ["Positive", "Negative", "Other"],
            "values": [s.totals["positive"], s.totals["negative"], s.totals["neutral"]]}


def build_exports(answer: Answer, summary_text: str = "") -> dict:
    """Generate export bytes on demand.

    Previously every turn eagerly built a PPTX, XLSX and CSV and stashed
    them in memory whether or not anyone clicked download — hundreds of
    milliseconds of CPU and megabytes of state per question.
    """
    meta = answer.export_meta or {}
    rows = answer.export_rows or []
    if not rows and not meta:
        return {}
    summary = meta.get("summary") or P.split_into_points(summary_text, max_points=4) or ["No summary available."]
    return {
        "csv": exports.build_csv_bytes(rows),
        "excel": exports.build_excel_bytes(rows),
        "pptx": exports.build_pptx_bytes(
            title=meta.get("title", "Voice of Grower"),
            subtitle=meta.get("subtitle", ""),
            summary_lines=summary,
            kpis=meta.get("kpis", {}),
            chart_title=meta.get("chart_title", ""),
            chart_labels=meta.get("chart_labels"),
            chart_values=meta.get("chart_values"),
            chart_type=meta.get("chart_type", "column"),
            table_headers=meta.get("table_headers"),
            table_rows=meta.get("table_rows"),
        ),
    }
