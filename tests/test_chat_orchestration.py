"""
End-to-end tests for process_chat_query / finalize_normal_response against a
fake Pinecone backend seeded with known records — these pin down the
observable behavior a user actually sees, not just internal helpers.
"""
from conftest import make_record
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


def test_finalize_normal_response_produces_chart_and_downloads(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Great results with Isabion."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery of Isabion."),
    ])
    state = vc.process_chat_query("what do growers think about isabion?", "fake-key")
    assert state["kind"] == "normal"

    result = vc.finalize_normal_response(state, "Growers are mostly satisfied, though delivery was slow.")
    assert result["chart"]["labels"] == ["Positive", "Negative", "Other"]
    assert result["chart"]["values"] == [1, 1, 0]
    assert result["downloads"]["csv"]
    assert result["downloads"]["excel"]
    assert result["downloads"]["pptx"]
    assert "Isabion" in result["final_reply"]
