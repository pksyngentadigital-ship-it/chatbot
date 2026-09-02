"""
End-to-end tests for process_chat_query / finalize_normal_response against a
fake Pinecone backend seeded with known records — these pin down the
observable behavior a user actually sees, not just internal helpers.
"""
import json

from conftest import make_record, raising_groq_factory
import legacy_api as vc


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
        make_record("January", "2026", "neutral", "Product Queries",
                    "Grower asked about pricing for Axial.", products="Axial"),
        make_record("January", "2026", "neutral", "Others",
                    "Unrelated note about Isabion.", products="Isabion"),
    ])
    state = vc.process_chat_query("which products receive the highest price inquiries?", "fake-key")
    # This phrasing has an explicit ranking subject ("which products") +
    # rank word ("highest") -> deterministic ranking path, not "normal".
    assert state["kind"] == "ranking"
    # ...and only the Product Queries record is in scope, so the one filed
    # under Others must not be counted.
    assert "Axial" in state["reply"]
    assert "Isabion" not in state["reply"]


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


def test_kaho_is_never_detected_as_a_product(fake_pinecone_factory):
    # Reported live: "Kaho" is a Syngenta podcast/campaign name, not a
    # product, but it was mistakenly in PRODUCT_LIST and genuinely appears
    # in real feedback text ("Kaho Syngenta Podcast"), so it must also be
    # excluded from the dynamic product-probe fallback or it gets
    # re-confirmed as a "product" anyway.
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback",
                     "The Kaho Syngenta Podcast was well received by listeners."),
    ])
    state = vc.process_chat_query("what do people think about kaho?", "fake-key")
    assert state.get("context", {}).get("product") != "kaho"


def test_delayed_is_never_detected_as_a_product(fake_pinecone_factory):
    # Found live against the real production dataset: "delayed" — one of
    # the Phase 4 intent-keyword-expansion words — appears verbatim in real
    # feedback text, so the dynamic product-probe fallback mistook it for a
    # product name on complaint queries about shipping delays.
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback",
                     "Shipment was delayed by two weeks."),
    ])
    state = vc.process_chat_query("growers are frustrated with delayed shipments", "fake-key")
    assert state.get("context", {}).get("product") != "delayed"


def test_sales_scoping_survives_llm_fallback_even_when_it_guesses_an_intent(fake_pinecone_factory, fake_groq_factory):
    # Found live: a pricing query already correctly routed to Product
    # Queries via SALES_KEYWORDS, but since that isn't "intent_explicit",
    # the Phase 7 LLM fallback still fired, guessed intent="suggestion",
    # and silently re-routed the query to the Suggestions category —
    # changing the badge from Product Inquiries to Suggestions.
    fake_groq_factory(json.dumps({"product": None, "crop": None, "intent": "suggestion"}))
    fake_pinecone_factory([
        make_record("January", "2026", "neutral", "Product Queries", "Grower asked about pricing for Axial."),
    ])
    state = vc.process_chat_query(
        "what are growers asking about pricing?", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state["badge"] == "💰 Product Inquiries"


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


def test_llm_rescues_intent_when_no_keyword_matched(fake_pinecone_factory, fake_groq_factory):
    # The most common misread: a phrasing no keyword list covers silently
    # becomes "sentiment". Here the question is about themes, so the
    # classifier should return "topics" rather than letting it default.
    fake_groq_factory(json.dumps({"product": None, "crop": None, "intent": "topics"}))
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Cropwise app is useful"),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Packaging damaged"),
    ])
    state = vc.process_chat_query(
        "what is dominating the conversation right now?", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state["query_intent"] == "topics", "an unmatched phrasing must not silently default to sentiment"


def test_llm_intent_is_still_validated_against_the_enum(fake_pinecone_factory, fake_groq_factory):
    fake_groq_factory(json.dumps({"product": None, "crop": None, "intent": "extremely-enthusiastic"}))
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Something good"),
    ])
    state = vc.process_chat_query(
        "give me a read on things for growers", "fake-key", groq_api_key="fake-groq-key"
    )
    # Falls back to the deterministic default rather than adopting an
    # invented category.
    assert state.get("query_intent") in ("sentiment", "complaint", "positive", "suggestion", "topics")


def test_deterministic_intent_is_never_overridden_by_the_llm(fake_pinecone_factory, fake_groq_factory):
    fake_groq_factory(json.dumps({"product": None, "crop": None, "intent": "positive"}))
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Late delivery."),
    ])
    state = vc.process_chat_query(
        "what are the complaints this month?", "fake-key", groq_api_key="fake-groq-key"
    )
    assert state["query_intent"] == "complaint", "an explicit keyword match must win"


# ── Correction/meta-feedback short-circuit ──

def test_correction_message_short_circuits_before_any_retrieval(fake_pinecone_factory, fake_groq_factory):
    # The exact live-reported bug: a correction message with no domain
    # keyword or clear intent used to fall through to the Phase 7 LLM
    # fallback, which could non-deterministically guess "complaint" and
    # dump an unscoped wall of unrelated complaints. It must now be caught
    # before reaching Pinecone or Groq at all.
    calls = []

    def _tracking(**kwargs):
        calls.append(1)
        return "{}"

    fake_groq_factory(_tracking)
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Unrelated complaint about pricing."),
    ])
    state = vc.process_chat_query("Kaho is not a product of syngenta", "fake-key", groq_api_key="fake-groq-key")
    assert state["kind"] == "meta_feedback"
    assert state["reply"] == vc.CORRECTION_ACK_REPLY
    assert calls == []


def test_correction_message_works_without_any_api_keys():
    # No Pinecone/Groq key needed at all — the short-circuit happens before
    # either is touched.
    state = vc.process_chat_query("that's wrong, please fix this", None)
    assert state["kind"] == "meta_feedback"


# ── "farmers" false-positive + genuine "topics" intent ──

def test_farmers_is_never_detected_as_a_product(fake_pinecone_factory):
    # Reported live: "farmers" is one of the most common words in the
    # entire dataset (it's a grower-feedback app), so the dynamic
    # product-probe fallback confirmed it as a "product" for any query
    # containing the word — the same false-positive class as
    # other/pricing/delayed/kaho, just far higher-frequency.
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Farmers were happy with the results."),
    ])
    state = vc.process_chat_query("what are farmers talking most about?", "fake-key")
    assert state.get("context", {}).get("product") != "farmers"


def test_topics_query_gets_its_own_intent_and_badge(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Growers happy with pricing this month."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Packaging was damaged on arrival."),
    ])
    state = vc.process_chat_query("what are farmers talking most about?", "fake-key")
    assert state["kind"] == "normal"
    assert state["query_intent"] == "topics"
    assert state["badge"] == "🗣️ Top Topics"
    assert "TOPICS/THEMES" in state["system_prompt"]
    assert "positive-vs-negative sentiment" in state["system_prompt"]


def test_topics_query_is_not_sentiment_filtered(fake_pinecone_factory):
    # A topics question needs the full breadth of feedback (both positive
    # and negative) to find genuine themes — it must not be silently
    # narrowed to only positive or only negative records.
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Growers happy with pricing this month."),
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Packaging was damaged on arrival."),
    ])
    state = vc.process_chat_query("what topics are most discussed this month?", "fake-key")
    assert len(state["positive_bullets"]) == 1
    assert len(state["negative_bullets"]) == 1


def test_topics_keyword_takes_priority_over_generic_sentiment_default(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "x"),
    ])
    state = vc.process_chat_query("what are the most talked about subjects?", "fake-key")
    assert state.get("query_intent") == "topics"


# ── Questions ABOUT the tool, not about the data ──

def test_capability_question_is_answered_not_refused():
    # "What can I ask you?" is the most natural first question a new user
    # has, and it carries no domain vocabulary — the guardrail used to
    # refuse it with "I cannot generate this response".
    for q in [
        "What can you generate for me?",
        "What can I ask you?",
        "what can you do",
        "give me some examples",
        "how does this work?",
    ]:
        state = vc.process_chat_query(q, None)
        assert state["kind"] == "capability", q
        assert "I cannot generate this response" not in state["reply"]


def test_capability_reply_describes_real_features():
    state = vc.process_chat_query("what can you do", None)
    reply = state["reply"].lower()
    for feature in ["sentiment", "complaint", "compare", "trend", "powerpoint"]:
        assert feature in reply, feature


def test_a_real_data_question_is_not_mistaken_for_a_capability_question():
    assert vc.detect_capability_question("what are the complaints about isabion") is False
    assert vc.detect_capability_question("which crop has the most complaints") is False


def test_reply_only_turns_get_curated_followups_not_invented_ones():
    """Observed live: with no retrieved data to read, the suggester filled
    the vacuum with products that do not exist ("HydroBoost", a "SoilSense
    sensor"). Reply-only turns take the curated list instead."""
    from vog import compose
    from vog.catalog import SUGGESTED_PROMPTS_QUICK
    from vog.plan import build_plan

    def _boom(*a, **k):
        raise AssertionError("the model must not be asked on a reply-only turn")

    plan = build_plan("what can you do for me?")
    for kind in ("reply", "no_data"):
        answer = compose.Answer(kind=kind, text="...")
        got = compose.followups(plan, answer, "...", "fake-key")
        assert got == list(SUGGESTED_PROMPTS_QUICK[:3])
        assert all(s in SUGGESTED_PROMPTS_QUICK for s in got)


def test_followups_fall_back_to_curated_when_the_model_returns_nothing(monkeypatch):
    from vog import compose, llm
    from vog.catalog import SUGGESTED_PROMPTS_QUICK
    from vog.plan import build_plan

    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: None)
    answer = compose.Answer(kind="prompt", text="")
    got = compose.followups(build_plan("what are the complaints?"), answer, "body", "fake-key")
    assert got == list(SUGGESTED_PROMPTS_QUICK[:3]), "an empty suggestion list leaves dead space in the UI"


def test_by_month_phrasing_routes_to_the_trend_path():
    """One of the app's own suggested prompts. It fell through to a
    single-period summary, and the model then reported that the data
    contained no monthly information."""
    from vog.plan import MODE_TREND, build_plan
    for q in ("show overall grower sentiment by month",
              "complaints per month",
              "monthly breakdown of feedback"):
        assert build_plan(q).mode == MODE_TREND, q
