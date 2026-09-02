"""
Regression tests for query-parsing/detection logic. Several of these encode
real bugs that were caught during live QA and fixed once — they exist to
make sure those exact failures can never silently come back.
"""
import legacy_api as vc


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


def test_pests_in_this_dataset_are_never_products():
    # "Jassid" was ranking as the single most-mentioned product in
    # production — 60 mentions — purely because it was missing from
    # DISEASE_PEST_TERMS.
    for text in [
        "Jassid attack in cotton crop",
        "Mealy bug and Jassid problem",
        "Pink Bollworm damage in cotton",
        "Anthracnose in Chilli crop",
    ]:
        assert vc.extract_product_mentions(text) == [], text


def test_agronomy_advice_words_are_never_products():
    assert vc.extract_product_mentions("Solution for Jassid and whitefly in cotton") == []
    assert vc.extract_product_mentions("Dosage and Application details needed") == []


def test_a_crop_is_never_tagged_as_a_product():
    assert vc.extract_product_mentions("Anthracnose in Chilli crop") == []
    products = vc.extract_product_mentions("Tilt applied on Rice gave good results")
    assert products == ["Tilt"], f"crop leaked into products: {products}"


def test_real_products_still_survive_the_stricter_filters():
    assert "Isabion" in vc.extract_product_mentions("Excellent results with Isabion on wheat")
    assert "Amistar Top" in vc.extract_product_mentions("Amistar Top gave good control")


# ── Authoritative products vs discovery candidates ──

def test_product_tag_holds_only_catalog_matches():
    # The capitalized-phrase heuristic put 12 of the live top-20 "products"
    # in as agronomic vocabulary: Abiotic, Early (Early Blight), Late,
    # Blossom, Flower, White (White Fly), Horse (Horse Gram) and Potash
    # (a fragment of Naya Potash).
    for text in [
        "Early Blight in Tomato crop",
        "Abiotic Stress observed in cotton",
        "Blossom drop reported by growers",
        "White Fly attack on cotton",
        "Horse Gram cultivation query",
    ]:
        assert vc.extract_product_mentions(text) == [], text


def test_catalog_products_are_still_tagged():
    assert vc.extract_product_mentions("Naya Potash gave good results") == ["Naya Potash"]
    assert vc.extract_product_mentions("Axial worked well") == ["Axial"]
    # Tag order is not meaningful — these are counted independently.
    assert set(vc.extract_product_mentions("Isabion Gold and Virtako applied")) == {"Isabion Gold", "Virtako"}


def test_uncatalogued_names_are_simply_not_products():
    # No guessing step remains: a name that is not in the price list is not
    # a product, full stop. It enters the system by being added to the
    # catalog and re-ingested.
    assert vc.extract_product_mentions("Growers liked Zynora very much") == []
    assert vc.extract_product_mentions("Some NewBrandX product was tried") == []


def test_no_heuristic_guessing_machinery_remains():
    assert not hasattr(vc, "extract_product_candidates")
    assert not hasattr(vc, "detect_product_dynamic")


# ── Product master (Syngenta Pakistan price list, 8-Jun-2026) ──

def test_catalog_has_no_duplicates_or_crop_clashes():
    assert len(vc.PRODUCT_LIST) == len(set(vc.PRODUCT_LIST)), "duplicate catalog entries"
    clash = set(vc.PRODUCT_LIST) & set(vc.CROP_LIST)
    assert not clash, f"a name cannot be both a product and a crop: {clash}"


def test_price_list_brands_are_recognized():
    # One per section of the price list.
    samples = {
        "Actara gave good control": "Actara",            # insecticide
        "Ally Max applied to wheat": "Ally Max",          # herbicide
        "Amistar Top sprayed": "Amistar Top",             # fungicide
        "Dynasty CST seed treatment": "Dynasty CST",      # seed care
        "Quantis improved stress tolerance": "Quantis",   # biostimulant
        "Naya S Urea delivered late": "Naya S Urea",      # fertilizer
        "Klerat WB used for rodents": "Klerat WB",        # public health
    }
    for text, expected in samples.items():
        assert expected in vc.extract_product_mentions(text), text


def test_variant_suffixes_are_uppercased_not_title_cased():
    assert vc.extract_product_mentions("AXIAL XL performed well") == ["Axial XL"]
    assert vc.extract_product_mentions("Naya SOP arrived") == ["Naya SOP"]


def test_newly_ambiguous_price_list_names_need_capitalization():
    # "Icon" and "Machete" are real brands but also ordinary words.
    assert vc.extract_product_mentions("the icon on the app is unclear") == []
    assert vc.extract_product_mentions("Icon 10 CS was applied") == ["Icon"]


def test_cropwise_survives_even_though_it_is_not_a_price_list_sku():
    # Cropwise is the grower app, not a crop-protection product, so it is
    # absent from the CP price list — but it is one of the most-discussed
    # subjects in the feedback and must still resolve.
    assert "Cropwise" in vc.extract_product_mentions("The Cropwise app is very useful")


def test_extract_product_mentions_excludes_syngenta_itself():
    text = "Syngenta representative visited the farm this week."
    mentions = vc.extract_product_mentions(text)
    assert "Syngenta" not in mentions


def test_extract_product_mentions_excludes_the_word_other():
    # "Other" shows up constantly in ordinary feedback prose and must never
    # be tagged as a product — this was a real production bug.
    text = "Other than that, growers had no other complaints this month."
    mentions = vc.extract_product_mentions(text)
    assert "Other" not in mentions
    assert "Others" not in mentions


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


def test_followup_reference_detects_more_insights_phrasing():
    assert vc.detect_followup_reference("what other insights can you tell me?") is True
    assert vc.detect_followup_reference("anything else you can share?") is True


# ── detect_wants_more ──

def test_wants_more_detects_explicit_more_request():
    assert vc.detect_wants_more("what other insights can you tell me?") is True
    assert vc.detect_wants_more("what else can you tell me about it?") is True


def test_wants_more_false_on_fresh_question():
    assert vc.detect_wants_more("what are the top complaints for wheat?") is False


# ── is_query_in_scope: the strict topic guardrail ──

def test_query_in_scope_true_for_domain_query():
    assert vc.is_query_in_scope("what is the sentiment for wheat growers?") is True


def test_query_in_scope_false_for_off_topic_query():
    assert vc.is_query_in_scope("write me a poem about the ocean") is False


# ── detect_correction_or_meta_feedback ──

def test_correction_detected_product_denial():
    assert vc.detect_correction_or_meta_feedback("kaho is not a product of syngenta") is True


def test_correction_detected_various_phrasings():
    assert vc.detect_correction_or_meta_feedback("that's wrong, please fix this") is True
    assert vc.detect_correction_or_meta_feedback("you're mistaken about that") is True
    assert vc.detect_correction_or_meta_feedback("stop treating pricing as a product") is True
    assert vc.detect_correction_or_meta_feedback("wheat is not a real crop") is True


def test_correction_false_on_normal_data_questions():
    assert vc.detect_correction_or_meta_feedback("what do growers think about isabion?") is False
    assert vc.detect_correction_or_meta_feedback("which crop generated the highest number of complaints?") is False
    assert vc.detect_correction_or_meta_feedback("show me the top 10 complaints this month") is False
