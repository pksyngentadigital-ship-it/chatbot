"""
Regression tests for query-parsing/detection logic. Several of these encode
real bugs that were caught during live QA and fixed once — they exist to
make sure those exact failures can never silently come back.
"""
import vog_core as vc


# ── detect_product_known / detect_crop ──

def test_detect_product_known_matches_catalog_product():
    assert vc.detect_product_known("what do growers say about isabion?") == "isabion"


def test_detect_product_known_respects_word_boundaries():
    # "score" must not fire on substrings like "scorecard"
    assert vc.detect_product_known("show me the scorecard for this month") is None


def test_detect_crop_matches_catalog_crop():
    assert vc.detect_crop("issues reported in wheat this year") == "wheat"


def test_detect_crop_no_false_positive_on_unrelated_text():
    assert vc.detect_crop("what are the top complaints overall?") is None


# ── extract_product_mentions: the "Excellent"/"days"/"reported" false-positive bug ──

def test_extract_product_mentions_known_catalog_hit():
    text = "Grower reported excellent results with Isabion on wheat."
    mentions = vc.extract_product_mentions(text)
    assert "Isabion" in mentions


def test_extract_product_mentions_excludes_generic_capitalized_words():
    # "Excellent", "However", "Add" must never be tagged as products —
    # this was the original ranking-pollution bug.
    text = "Excellent. However, growers reported the results were good. Add more stock."
    mentions = vc.extract_product_mentions(text)
    for bad in ("Excellent", "However", "Add", "Good", "Results"):
        assert bad not in mentions


def test_extract_product_mentions_excludes_diseases_and_pests():
    text = "Septoria control with Score was effective on wheat."
    mentions = vc.extract_product_mentions(text)
    assert "Septoria" not in mentions
    assert "Score" in mentions


def test_extract_product_mentions_excludes_syngenta_itself():
    text = "Syngenta representative visited the farm this week."
    mentions = vc.extract_product_mentions(text)
    assert "Syngenta" not in mentions


def test_extract_product_mentions_no_duplicates():
    text = "Isabion worked well. Isabion Gold was also appreciated."
    mentions = vc.extract_product_mentions(text)
    assert len(mentions) == len(set(m.lower() for m in mentions))


# ── extract_crops ──

def test_extract_crops_finds_multiple_known_crops():
    text = "Feedback covers both wheat and cotton growers."
    crops = vc.extract_crops(text)
    assert set(crops) == {"Wheat", "Cotton"}


def test_extract_crops_empty_when_none_mentioned():
    assert vc.extract_crops("General product feedback with no crop named.") == []


# ── extract_all_months / extract_all_years / extract_all_weeks ──

def test_extract_all_months_dedupes_and_preserves_order():
    months = vc.extract_all_months("compare february and january, also feb again")
    assert months == ["February", "January"]


def test_extract_all_years_finds_multiple():
    years = vc.extract_all_years("compare 2025 and 2026 data")
    assert years == ["2025", "2026"]


def test_extract_all_weeks_normalizes_ordinal_words():
    weeks = vc.extract_all_weeks("compare the first week and 3rd week")
    assert weeks == ["1st", "3rd"]


# ── detect_aggregation_request: the intent-misrouting bug ──

def test_aggregation_request_detects_explicit_product_ranking():
    assert vc.detect_aggregation_request("which product received the highest number of complaints?") == "product"


def test_aggregation_request_detects_explicit_crop_ranking():
    assert vc.detect_aggregation_request("which crop generated the highest number of complaints?") == "crop"


def test_aggregation_request_top_n_phrasing():
    assert vc.detect_aggregation_request("show the top 5 products by complaint frequency") == "product"


def test_aggregation_request_does_not_fire_on_synthesis_asks():
    # This is the exact query that used to be misrouted into deterministic
    # ranking instead of LLM synthesis.
    assert vc.detect_aggregation_request("what are the most common product improvement recommendations?") is None


def test_aggregation_request_does_not_fire_on_bare_cooccurrence():
    assert vc.detect_aggregation_request("tell me about the product this month") is None


def test_aggregation_request_none_without_rank_phrase():
    assert vc.detect_aggregation_request("which products are available for wheat?") is None


# ── detect_trend_request ──

def test_trend_request_detects_explicit_trend_phrasing():
    assert vc.detect_trend_request("show the monthly trend for complaints") is True


def test_trend_request_ignores_bare_common_words():
    assert vc.detect_trend_request("what are the sales for our products?") is False


# ── detect_output_format ──

def test_output_format_excel_takes_priority_signal():
    assert vc.detect_output_format("export this to excel please") == "excel"


def test_output_format_ppt():
    assert vc.detect_output_format("give me a powerpoint-ready summary") == "ppt"


def test_output_format_table():
    assert vc.detect_output_format("show this as a table") == "table"


def test_output_format_chart():
    assert vc.detect_output_format("visualize this as a chart") == "chart"


def test_output_format_none_when_unspecified():
    assert vc.detect_output_format("what do growers think about isabion?") is None


# ── detect_followup_reference: narrow, explicit-phrase gated ──

def test_followup_reference_detects_explicit_continuation():
    assert vc.detect_followup_reference("what about wheat?") is True
    assert vc.detect_followup_reference("and for isabion?") is True


def test_followup_reference_false_on_unrelated_fresh_question():
    assert vc.detect_followup_reference("what are the top complaints for cotton?") is False


# ── is_query_in_scope: the strict topic guardrail ──

def test_query_in_scope_true_for_domain_query():
    assert vc.is_query_in_scope("what is the sentiment for wheat growers?") is True


def test_query_in_scope_false_for_off_topic_query():
    assert vc.is_query_in_scope("write me a poem about the ocean") is False
