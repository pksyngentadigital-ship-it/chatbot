"""
Compound-question handling.

Comparison previously worked along ONE axis only: time. Comparing two
products or two crops was not implemented at all, and detection returned
just the first catalog match — scanning in CATALOG order, not query order.
So "Compare customer sentiment for Tilt and Isabion", which ships as one
of the app's own suggested prompts, silently answered about Isabion alone.
"""
from conftest import make_record
import vog_core as vc


# ── Multi-subject detection ──

def test_all_products_in_the_query_are_detected_in_the_order_written():
    assert vc.detect_all_products("compare customer sentiment for tilt and isabion") == ["tilt", "isabion"]
    assert vc.detect_all_products("isabion versus tilt") == ["isabion", "tilt"]


def test_longer_product_name_wins_over_its_own_prefix():
    products = vc.detect_all_products("how is isabion gold performing")
    assert products == ["isabion gold"], f"prefix leaked: {products}"


def test_all_crops_detected_and_synonyms_collapsed():
    assert vc.detect_all_crops("compare wheat and cotton") == ["wheat", "cotton"]
    # rice and paddy are the same crop, so this is ONE subject, not two.
    assert len(vc.detect_all_crops("rice and paddy yields")) == 1


def test_single_product_query_is_unchanged():
    assert vc.detect_product_known("what about isabion") == "isabion"
    assert vc.detect_product_known("no products here") is None


# ── Product-vs-product comparison ──

def _two_product_index(fake_pinecone_factory):
    records = []
    for i in range(6):
        records.append(make_record("January", "2026", "positive", "Positive Feedback",
                                   f"Isabion gave excellent results {i}"))
    for i in range(6):
        records.append(make_record("January", "2026", "negative", "Complaint/Negative Feedback",
                                   f"Tilt caused leaf burn {i}"))
    return fake_pinecone_factory(records)


def test_two_products_produce_a_side_by_side_comparison(fake_pinecone_factory):
    _two_product_index(fake_pinecone_factory)
    state = vc.process_chat_query("Compare customer sentiment for Tilt and Isabion", "fake-key")

    assert state["kind"] == "normal"
    labels = [lbl for lbl, *_ in state["period_results"]]
    assert labels == ["Tilt", "Isabion"], f"both products must be sections, got {labels}"


def test_each_side_of_the_comparison_is_filtered_to_its_own_subject(fake_pinecone_factory):
    _two_product_index(fake_pinecone_factory)
    state = vc.process_chat_query("Compare customer sentiment for Tilt and Isabion", "fake-key")

    for label, pos, neg, neut in state["period_results"]:
        for bullet in pos + neg + neut:
            assert label.lower() in bullet.lower(), \
                f"section {label!r} contained a bullet about something else: {bullet!r}"


def test_comparison_prompt_uses_subject_wording_not_period_wording(fake_pinecone_factory):
    _two_product_index(fake_pinecone_factory)
    state = vc.process_chat_query("Compare customer sentiment for Tilt and Isabion", "fake-key")
    prompt = state["system_prompt"]
    assert "side-by-side COMPARISON" in prompt
    assert "COMPARISON request across these periods" not in prompt
    assert "do not answer about only one" in prompt


def test_comparison_header_does_not_also_scope_to_one_subject(fake_pinecone_factory):
    _two_product_index(fake_pinecone_factory)
    state = vc.process_chat_query("Compare customer sentiment for Tilt and Isabion", "fake-key")
    header = state["header"]
    assert "Tilt" in header and "Isabion" in header
    # Would read "Isabion — Sentiment Comparison: Tilt vs Isabion" otherwise.
    assert not header.strip().startswith("🌾 Isabion")


def test_two_crops_also_compare(fake_pinecone_factory):
    records = [
        make_record("January", "2026", "positive", "Positive Feedback", f"Wheat did well {i}") for i in range(4)
    ] + [
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", f"Cotton had problems {i}") for i in range(4)
    ]
    fake_pinecone_factory(records)
    state = vc.process_chat_query("Compare grower feedback for wheat and cotton", "fake-key")
    labels = [lbl for lbl, *_ in (state.get("period_results") or [])]
    assert labels == ["Wheat", "Cotton"]


def test_time_comparison_still_wins_when_both_are_present(fake_pinecone_factory):
    # An explicit two-period question stays a TIME comparison even if it
    # also names two products.
    records = [
        make_record("January", "2026", "positive", "Positive Feedback", "Isabion fine"),
        make_record("February", "2026", "positive", "Positive Feedback", "Tilt fine"),
    ]
    fake_pinecone_factory(records)
    state = vc.process_chat_query("compare isabion and tilt between January 2026 and February 2026", "fake-key")
    labels = [lbl for lbl, *_ in (state.get("period_results") or [])]
    assert any("January" in l for l in labels) and any("February" in l for l in labels)


def test_single_product_question_is_not_turned_into_a_comparison(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "positive", "Positive Feedback", "Isabion gave excellent results"),
    ])
    state = vc.process_chat_query("what do growers think about isabion?", "fake-key")
    assert state["badge"] != "🔀 Comparison"
