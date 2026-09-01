"""
End-to-end tests for process_chat_query / finalize_normal_response against a
fake Pinecone backend seeded with known records — these pin down the
observable behavior a user actually sees, not just internal helpers.
"""
import json

from conftest import make_record, raising_groq_factory
import vog_core as vc


def test_off_topic_query_is_blocked(fake_pinecone_factory):
    fake_pinecone_factory([])
    state = vc.process_chat_query("write me a poem about the ocean", "fake-key")
    assert state["kind"] == "blocked"


def test_missing_pinecone_key_short_circuits():
    state = vc.process_chat_query("what is the sentiment for wheat?", None)
    assert state["kind"] == "no_key"


def test_no_data_found_for_scoped_product(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Axial on cotton."),
    ])
    state = vc.process_chat_query("what do growers say about isabion?", "fake-key")
    assert state["kind"] == "no_data"
    assert state["context"]["product"] == "isabion"


def test_normal_query_returns_prompts_for_llm(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion on wheat."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery of Isabion."),
    ])
    state = vc.process_chat_query("what do growers think about isabion?", "fake-key")
    assert state["kind"] == "normal"
    assert "Isabion" in state["system_prompt"]
    assert state["actual_point_count"] == 2
    assert state["context"] == {"product": "isabion", "crop": None, "intent": "sentiment"}


def test_complaint_intent_filters_out_positive_bullets(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery of Isabion."),
    ])
    state = vc.process_chat_query("what are the complaints about isabion?", "fake-key")
    assert state["query_intent"] == "complaint"
    assert state["positive_bullets"] == []
    assert len(state["negative_bullets"]) == 1


def test_sales_query_scopes_to_product_query_category(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "neutral", "Product Queries", "Grower asked about pricing for Axial."),
        make_record("January", "2026", "neutral", "Others", "Unrelated general note."),
    ])
    state = vc.process_chat_query("which products receive the highest price inquiries?", "fake-key")
    # This phrasing has an explicit ranking subject ("which products") +
    # rank word ("highest") -> deterministic ranking path, not "normal".
    assert state["kind"] == "ranking"


def test_ranking_query_counts_exactly(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "x", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "y", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "z", crop="Cotton"),
    ])
    state = vc.process_chat_query("which crop generated the highest number of complaints?", "fake-key")
    assert state["kind"] == "ranking"
    assert "Wheat" in state["reply"]
    # Exact count must appear verbatim — never left to an LLM to eyeball.
    assert "2 mentions" in state["reply"]


def test_synthesis_style_query_does_not_use_ranking_path(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "neutral", "Suggestions", "Growers suggested better packaging for Isabion."),
    ])
    state = vc.process_chat_query("what are the most common product improvement recommendations?", "fake-key")
    assert state["kind"] == "normal"


def test_trend_query_returns_chronological_table(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "a"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "b"),
        make_record("February", "2026", "negative", "Complaint/Negative Feedback", "c"),
    ])
    state = vc.process_chat_query("show the monthly trend for complaints", "fake-key")
    assert state["kind"] == "trend"
    assert "January 2026" in state["reply"]
    assert "February 2026" in state["reply"]


def test_followup_reference_inherits_prior_product_when_gap_left_unspecified(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion on wheat."),
        make_record("January", "2026", "positive", "Positive Feedback", "Growers liked Axial on cotton too."),
    ])
    prior_context = {"product": "isabion", "crop": None, "intent": "sentiment"}
    state = vc.process_chat_query("what about last month?", "fake-key", prior_context=prior_context)
    # "what about" is a follow-up phrase and the query itself names no
    # product, so it must inherit "isabion" from prior context.
    assert state.get("context", {}).get("product") == "isabion"


def test_followup_inheritance_never_overrides_explicit_current_query_subject(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Axial on cotton."),
    ])
    prior_context = {"product": "isabion", "crop": None, "intent": "sentiment"}
    # Explicitly names Axial -> must NOT be overridden by prior "isabion",
    # even though this also contains a follow-up phrase.
    state = vc.process_chat_query("what about axial?", "fake-key", prior_context=prior_context)
    assert state.get("context", {}).get("product") == "axial"


def test_non_followup_fresh_question_ignores_prior_context(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Axial on cotton."),
    ])
    prior_context = {"product": "isabion", "crop": None, "intent": "sentiment"}
    # No follow-up phrase -> a fresh, unrelated question must never inherit
    # stale product scoping from a previous turn.
    state = vc.process_chat_query("what are the top complaints this month?", "fake-key", prior_context=prior_context)
    assert state.get("context", {}).get("product") is None


def test_other_is_never_detected_as_a_product(fake_pinecone_factory):
    # Real feedback text often contains the word "other" (e.g. "no other
    # issues") — the dynamic product-probe fallback must not latch onto it.
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback",
                     "Growers had no other complaints besides minor packaging issues."),
    ])
    state = vc.process_chat_query("what other feedback do you have?", "fake-key")
    assert state.get("context", {}).get("product") != "other"


def test_pricing_is_never_detected_as_a_product(fake_pinecone_factory):
    # Same bug class as "other": "pricing" is a real English word that
    # shows up verbatim in feedback text, so the dynamic product-probe
    # fallback was mistaking it for a product name on sales/pricing queries.
    fake_pinecone_factory([
        make_record("January", "2026", "neutral", "Product Queries", "Grower asked about pricing for Axial."),
    ])
    state = vc.process_chat_query("what are growers asking about pricing?", "fake-key")
    assert state.get("context", {}).get("product") != "pricing"


def test_wants_more_followup_passes_prior_reply_to_avoid_repetition(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion on wheat."),
        make_record("January", "2026", "positive", "Positive Feedback", "Growers liked the packaging too."),
    ])
    prior_context = {
        "product": "isabion", "crop": None, "intent": "sentiment",
        "last_reply": "Growers are happy with Isabion's performance on wheat.",
    }
    state = vc.process_chat_query("what other insights can you tell me?", "fake-key", prior_context=prior_context)
    assert state["kind"] == "normal"
    # Still scoped to the inherited product...
    assert state["context"]["product"] == "isabion"
    # ...and the LLM is told what was already said, so it doesn't repeat it.
    assert "CONTINUATION REQUEST" in state["system_prompt"]
    assert "Growers are happy with Isabion's performance on wheat." in state["system_prompt"]


def test_plain_followup_does_not_inject_continuation_clause(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion on wheat."),
    ])
    prior_context = {"product": "isabion", "crop": None, "intent": "sentiment", "last_reply": "Some earlier reply."}
    # "what about wheat?" is a follow-up but NOT a "give me more" request —
    # the continuation clause should only fire for explicit more-insights phrasing.
    state = vc.process_chat_query("what about wheat?", "fake-key", prior_context=prior_context)
    assert "CONTINUATION REQUEST" not in state["system_prompt"]


def test_finalize_normal_response_omits_chart_when_not_requested(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery of Isabion."),
    ])
    state = vc.process_chat_query("what do growers think about isabion?", "fake-key")
    assert state["kind"] == "normal"

    result = vc.finalize_normal_response(state, "Growers are mostly satisfied, though delivery was slow.")
    # Plain question, no "chart"/"visualize" wording -> no chart shown inline.
    assert result["chart"] is None
    # Downloads are opt-in via an explicit click, so they keep their chart regardless.
    assert result["downloads"]["csv"]
    assert result["downloads"]["excel"]
    assert result["downloads"]["pptx"]
    assert "Isabion" in result["final_reply"]


def test_finalize_normal_response_includes_chart_when_explicitly_requested(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery of Isabion."),
    ])
    state = vc.process_chat_query("show me a chart of sentiment for isabion", "fake-key")
    assert state["kind"] == "normal"
    assert state["output_format"] == "chart"

    result = vc.finalize_normal_response(state, "Growers are mostly satisfied, though delivery was slow.")
    assert result["chart"]["labels"] == ["Positive", "Negative", "Other"]
    assert result["chart"]["values"] == [1, 1, 0]


def test_ranking_chart_omitted_by_default(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "x", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "y", crop="Cotton"),
    ])
    state = vc.process_chat_query("which crop generated the highest number of complaints?", "fake-key")
    assert state["kind"] == "ranking"
    assert state["chart"] is None
    # Downloads (PPTX/Excel) keep their chart regardless of inline display.
    assert state["downloads"]["pptx"]


def test_ranking_chart_included_when_explicitly_requested(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "x", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "y", crop="Cotton"),
    ])
    state = vc.process_chat_query("show a chart of which crop generated the highest number of complaints", "fake-key")
    assert state["kind"] == "ranking"
    assert state["chart"] is not None
    assert state["chart"]["labels"]


def test_trend_chart_omitted_by_default(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "a"),
        make_record("February", "2026", "negative", "Complaint/Negative Feedback", "b"),
    ])
    state = vc.process_chat_query("show the monthly trend for complaints", "fake-key")
    assert state["kind"] == "trend"
    assert state["chart"] is None


def test_trend_chart_included_when_explicitly_requested(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "a"),
        make_record("February", "2026", "negative", "Complaint/Negative Feedback", "b"),
    ])
    state = vc.process_chat_query("show a chart of the monthly trend for complaints", "fake-key")
    assert state["kind"] == "trend"
    assert state["chart"] is not None


# ── Sales/pricing badge fix (Phase 4) ──

def test_sales_pricing_query_gets_product_inquiries_badge(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "neutral", "Product Queries", "Grower asked about pricing for Axial."),
    ])
    state = vc.process_chat_query("what are growers asking about pricing?", "fake-key")
    assert state["kind"] == "normal"
    assert state["badge"] == "💰 Product Inquiries"


# ── Expanded intent keyword coverage (Phase 4) ──

def test_expanded_complaint_keywords_catch_real_phrasing(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Shipment was delayed."),
    ])
    state = vc.process_chat_query("growers are frustrated with delayed shipments", "fake-key")
    assert state.get("query_intent") == "complaint"


def test_expanded_positive_keywords_catch_real_phrasing(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results."),
    ])
    state = vc.process_chat_query("growers are really happy and impressed with the results", "fake-key")
    assert state.get("query_intent") == "positive"


def test_expanded_suggestion_keywords_catch_real_phrasing(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "neutral", "Suggestions", "Better packaging please."),
    ])
    state = vc.process_chat_query("growers would like better packaging", "fake-key")
    assert state.get("query_intent") == "suggestion"


# ── Relative time-window end-to-end (Phase 3) ──

def test_relative_window_last_n_months_merges_across_months(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion."),
        make_record("February", "2026", "positive", "Positive Feedback", "Isabion worked well again."),
        make_record("November", "2025", "positive", "Positive Feedback", "Too old to be in the window."),
    ])
    state = vc.process_chat_query("what do growers think about isabion in the last 2 months?", "fake-key")
    assert state["kind"] == "normal"
    assert state["actual_point_count"] == 2
    assert "the last 2 months" in state["timeframe_label"]


def test_relative_window_no_data_falls_back_gracefully(fake_pinecone_factory):
    fake_pinecone_factory([])
    state = vc.process_chat_query("show grower feedback for the last 3 months", "fake-key")
    # No dated records at all in the index -> resolve_relative_window
    # returns None -> falls through to the plain (also empty) path,
    # never crashes.
    assert state["kind"] == "no_data"


# ── Ranking/trend narrative synthesis (Phase 6) ──

def test_ranking_narrative_appended_when_groq_available(fake_pinecone_factory, fake_groq_factory):
    fake_groq_factory("Wheat had more disease-related complaints this season.")
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "disease issue", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "disease issue 2", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "z", crop="Cotton"),
    ])
    state = vc.process_chat_query(
        "which crop generated the highest number of complaints?", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state["kind"] == "ranking"
    assert "Wheat had more disease-related complaints this season." in state["reply"]


def test_ranking_narrative_absent_without_groq_key(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "x", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "y", crop="Wheat"),
    ])
    state = vc.process_chat_query("which crop generated the highest number of complaints?", "fake-key")
    assert state["kind"] == "ranking"
    assert "*Why" not in state["reply"]


def test_ranking_narrative_failure_does_not_break_numbers(fake_pinecone_factory, monkeypatch):
    raising_groq_factory(monkeypatch)
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "x", crop="Wheat"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "y", crop="Wheat"),
    ])
    state = vc.process_chat_query(
        "which crop generated the highest number of complaints?", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state["kind"] == "ranking"
    assert "Wheat" in state["reply"]
    assert "2 mentions" in state["reply"]


# ── LLM-assisted query understanding fallback (Phase 7) ──

def test_llm_fallback_resolves_ambiguous_query_when_regex_finds_nothing(fake_pinecone_factory, fake_groq_factory):
    fake_groq_factory(json.dumps({"product": "isabion", "crop": None, "intent": "complaint"}))
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery of Isabion."),
    ])
    # Deliberately vague — contains "product" (passes the topic guardrail)
    # but no specific product/crop/intent keyword the regex layer recognizes.
    state = vc.process_chat_query(
        "tell me about the biologicals product line", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state.get("context", {}).get("product") == "isabion"


def test_llm_fallback_never_overrides_explicit_regex_detection(fake_pinecone_factory, fake_groq_factory):
    # Even if the LLM (implausibly) gets called, it must never override an
    # already-resolved product from a query that explicitly names one.
    fake_groq_factory(json.dumps({"product": "axial", "crop": None, "intent": "positive"}))
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion."),
    ])
    state = vc.process_chat_query(
        "what do growers think about isabion?", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state.get("context", {}).get("product") == "isabion"


def test_llm_fallback_not_invoked_when_intent_already_explicit(fake_pinecone_factory, fake_groq_factory):
    calls = []

    def _tracking(**kwargs):
        calls.append(1)
        return "{}"

    fake_groq_factory(_tracking)
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery."),
    ])
    vc.process_chat_query("what are the complaints this month?", "fake-key", groq_api_key="fake-groq-key")
    assert calls == []
