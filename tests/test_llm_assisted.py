"""
Regression tests for the LLM-assisted features (Phases 5-7): follow-up
suggestions, ranking/trend narrative synthesis, and the validated
query-understanding fallback. The two non-negotiable properties tested
throughout: (1) a broken or misbehaving LLM call must never break the main
deterministic answer, and (2) any LLM-proposed product/crop must be
validated against the real catalog before being trusted — the exact check
that would have caught a live-observed hallucination ("Price Drop
Herbicide") before it ever reached a user.
"""
import json

from conftest import raising_groq_factory
import legacy_api as vc


# ── generate_followup_suggestions ──

def test_followup_suggestions_parses_json_array(fake_groq_factory):
    fake_groq_factory(json.dumps(["What about wheat?", "Show this as a trend?", "Compare to last year?"]))
    suggestions = vc.generate_followup_suggestions("sentiment", "Isabion", "January 2026", "Growers are happy.", "fake-key")
    assert suggestions == ["What about wheat?", "Show this as a trend?", "Compare to last year?"]


def test_followup_suggestions_strips_code_fence(fake_groq_factory):
    fake_groq_factory('```json\n["One question?"]\n```')
    suggestions = vc.generate_followup_suggestions("sentiment", None, "January 2026", "reply", "fake-key")
    assert suggestions == ["One question?"]


def test_followup_suggestions_empty_on_malformed_json(fake_groq_factory):
    fake_groq_factory("not valid json at all")
    assert vc.generate_followup_suggestions("sentiment", None, "January 2026", "reply", "fake-key") == []


def test_followup_suggestions_empty_on_groq_failure(monkeypatch):
    raising_groq_factory(monkeypatch)
    assert vc.generate_followup_suggestions("sentiment", None, "January 2026", "reply", "fake-key") == []


def test_followup_suggestions_empty_without_api_key():
    assert vc.generate_followup_suggestions("sentiment", None, "January 2026", "reply", None) == []


def test_followup_suggestions_respects_max_count(fake_groq_factory):
    fake_groq_factory(json.dumps(["a", "b", "c", "d", "e"]))
    suggestions = vc.generate_followup_suggestions("sentiment", None, "January 2026", "reply", "fake-key", max_suggestions=3)
    assert len(suggestions) == 3


# ── generate_deterministic_narrative ──

def test_narrative_grounded_in_bullets(fake_groq_factory):
    fake_groq_factory("Growers praised fast results.")
    narrative = vc.generate_deterministic_narrative("product", "Isabion", 5, ["Great results with Isabion"], "fake-key")
    assert narrative == "Growers praised fast results."


def test_narrative_empty_without_bullets(fake_groq_factory):
    fake_groq_factory("should never be called")
    assert vc.generate_deterministic_narrative("product", "Isabion", 5, [], "fake-key") == ""


def test_narrative_empty_on_groq_failure(monkeypatch):
    raising_groq_factory(monkeypatch)
    assert vc.generate_deterministic_narrative("product", "Isabion", 5, ["some bullet"], "fake-key") == ""


def test_narrative_empty_without_api_key():
    assert vc.generate_deterministic_narrative("product", "Isabion", 5, ["some bullet"], None) == ""


# ── llm_assisted_query_understanding: the critical validation gate ──

def test_llm_query_understanding_accepts_known_product(fake_groq_factory):
    fake_groq_factory(json.dumps({"product": "isabion", "crop": None, "intent": "complaint"}))
    result = vc.llm_assisted_query_understanding("how's it going with the biologicals line", "fake-key")
    assert result == {"product": "isabion", "crop": None, "intent": "complaint"}


def test_llm_query_understanding_accepts_topics_intent(fake_groq_factory):
    fake_groq_factory(json.dumps({"product": None, "crop": None, "intent": "topics"}))
    result = vc.llm_assisted_query_understanding("what's everyone talking about lately", "fake-key")
    assert result == {"product": None, "crop": None, "intent": "topics"}


def test_llm_query_understanding_rejects_hallucinated_product(fake_groq_factory):
    # The exact failure mode observed live: the LLM invents a plausible-
    # sounding product name that isn't in the real catalog.
    fake_groq_factory(json.dumps({"product": "price drop herbicide", "crop": None, "intent": "sentiment"}))
    result = vc.llm_assisted_query_understanding("what's going on with pricing", "fake-key")
    assert result == {"product": None, "crop": None, "intent": "sentiment"}


def test_llm_query_understanding_rejects_hallucinated_crop(fake_groq_factory):
    fake_groq_factory(json.dumps({"product": None, "crop": "unicorn fruit", "intent": None}))
    assert vc.llm_assisted_query_understanding("vague query", "fake-key") is None


def test_llm_query_understanding_rejects_out_of_enum_intent(fake_groq_factory):
    fake_groq_factory(json.dumps({"product": "isabion", "crop": None, "intent": "excited"}))
    result = vc.llm_assisted_query_understanding("vague query", "fake-key")
    assert result == {"product": "isabion", "crop": None, "intent": None}


def test_llm_query_understanding_none_on_all_null(fake_groq_factory):
    fake_groq_factory(json.dumps({"product": None, "crop": None, "intent": None}))
    assert vc.llm_assisted_query_understanding("vague query", "fake-key") is None


def test_llm_query_understanding_none_on_malformed_json(fake_groq_factory):
    fake_groq_factory("not json")
    assert vc.llm_assisted_query_understanding("vague query", "fake-key") is None


def test_llm_query_understanding_none_on_groq_failure(monkeypatch):
    raising_groq_factory(monkeypatch)
    assert vc.llm_assisted_query_understanding("vague query", "fake-key") is None


def test_llm_query_understanding_none_without_api_key():
    assert vc.llm_assisted_query_understanding("vague query", None) is None
