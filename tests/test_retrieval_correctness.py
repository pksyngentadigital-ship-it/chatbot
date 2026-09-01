"""
Retrieval correctness tests (Stage 4).

The headline case is test_exported_kpis_report_true_counts: the app's whole
premise is that its numbers are exact, and the exported KPIs were being
computed from the truncated slice sent to the model rather than the real
match set — so a month with 60 negative records exported "Negative: 12"
into a PowerPoint.
"""
from conftest import make_record
import vog_core as vc


# ── Truthful counts ──

def test_exported_kpis_report_true_counts_not_the_truncated_sample(fake_pinecone_factory):
    records = [make_record("January", "2026", "negative", "Complaint/Negative Feedback", f"neg {i}") for i in range(60)]
    records += [make_record("January", "2026", "positive", "Positive Feedback", f"pos {i}") for i in range(5)]
    fake_pinecone_factory(records)

    state = vc.process_chat_query("show overall grower sentiment for January 2026", "fake-key")
    result = vc.finalize_normal_response(state, "Summary text.")

    assert result["kpis"]["Negative"] == 60, "KPI must be the real count, not the truncated slice"
    assert result["kpis"]["Positive"] == 5
    assert result["kpis"]["Total matching records"] == 65


def test_prompt_declares_a_sample_when_data_was_truncated(fake_pinecone_factory):
    records = [make_record("January", "2026", "negative", "Complaint/Negative Feedback", f"neg {i}") for i in range(60)]
    fake_pinecone_factory(records)
    state = vc.process_chat_query("what are the complaints in January 2026?", "fake-key")
    assert "a sample, NOT the complete set" in state["user_prompt"]
    assert "do not state or imply totals" in state["user_prompt"]


def test_small_result_set_is_still_described_as_a_total(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "only complaint"),
    ])
    state = vc.process_chat_query("what are the complaints in January 2026?", "fake-key")
    assert "total" in state["user_prompt"]
    assert "a sample, NOT the complete set" not in state["user_prompt"]


# ── The question must survive retrieval ──

def test_subject_boost_blends_with_the_query_rather_than_replacing_it():
    query = [1.0, 0.0, 0.0, 0.0]
    subject = [0.0, 1.0, 0.0, 0.0]
    blended = vc._blend_vectors(query, subject, vc.SUBJECT_BLEND_WEIGHT)

    assert blended != query, "the subject must actually influence retrieval"
    assert blended[1] > 0, "the subject component must be present"
    assert blended[0] > 0, "the user's question must NOT be discarded"
    assert blended[0] > blended[1], "the question should still outweigh the subject boost"
    assert abs(sum(v * v for v in blended) ** 0.5 - 1.0) < 1e-9, "must be unit length"


def test_blend_weight_keeps_the_question_dominant():
    # Above ~0.5 the question stops mattering, which is the bug this fixes.
    assert 0 < vc.SUBJECT_BLEND_WEIGHT < 0.5


def test_blend_degrades_safely_on_bad_input():
    q = [1.0, 0.0]
    assert vc._blend_vectors(q, [], 0.5) == q
    assert vc._blend_vectors(q, [1.0, 2.0, 3.0], 0.5) == q, "length mismatch must fall back"
    assert vc._blend_vectors([0.0, 0.0], [0.0, 0.0], 0.5) == [0.0, 0.0], "zero vector must not divide by zero"


def test_prompt_instructs_the_model_to_answer_the_actual_question(fake_pinecone_factory):
    fake_pinecone_factory([
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", "Isabion packaging leaks."),
    ])
    state = vc.process_chat_query("what packaging problems does isabion have?", "fake-key")
    prompt = state["system_prompt"]
    assert "ANSWER THE QUESTION THAT WAS ASKED" in prompt
    assert "Do NOT substitute a general overview" in prompt


# ── Word-boundary filtering ──

def test_crop_filter_does_not_match_inside_other_words():
    bullets = [
        "Complaint/Negative Feedback: The price of the product is too high.",
        "Positive Feedback: Rice yield improved significantly.",
        "Suggestions: The loyalty program should be simplified.",
    ]
    assert vc.filter_bullets_by_crop(bullets, "rice") == [bullets[1]], "'rice' must not match 'price'"
    assert vc.filter_bullets_by_crop(bullets, "gram") == [], "'gram' must not match 'program'"


def test_crop_filter_matches_synonyms():
    assert vc.filter_bullets_by_crop(["Positive: paddy did well"], "rice"), \
        "a rice query should also match bullets that say paddy"


def test_product_filter_is_word_bounded():
    bullets = ["Positive: the scorecard looked fine", "Positive: Score worked well"]
    assert vc.filter_bullets_by_product(bullets, "score") == [bullets[1]]


# ── Breadth for breadth-seeking intents ──

def test_topics_intent_gets_a_larger_sample_than_a_product_lookup(fake_pinecone_factory):
    records = [make_record("January", "2026", "positive", "Positive Feedback", f"point {i}") for i in range(40)]
    fake_pinecone_factory(records)
    state = vc.process_chat_query("what are growers talking most about?", "fake-key")
    assert state["actual_point_count"] > 12, \
        "a themes question needs breadth; 12 records is a single-product sample size"


# ── top-N ──

def test_requested_top_n_is_parsed():
    assert vc.detect_requested_top_n("show the top 5 products") == 5
    assert vc.detect_requested_top_n("top five crops by complaints") == 5
    assert vc.detect_requested_top_n("which crop has the most complaints") is None


def test_ranking_honours_the_requested_top_n(fake_pinecone_factory):
    records = [
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", f"x{i}", crop=f"Crop{i}")
        for i in range(12)
    ]
    fake_pinecone_factory(records)
    state = vc.process_chat_query("show the top 3 crops with the highest number of complaints", "fake-key")
    assert state["kind"] == "ranking"
    data_rows = [ln for ln in state["reply"].splitlines() if ln.startswith("| ") and "| Rank |" not in ln]
    assert len(data_rows) == 3, f"asked for top 3, got {len(data_rows)} rows"


# ── Month-over-month honesty ──

def test_densify_inserts_missing_months_so_mom_is_actually_month_over_month():
    dense = vc.densify_monthly_counts([("November 2025", 20), ("February 2026", 80)])
    assert [d[0] for d in dense] == ["November 2025", "December 2025", "January 2026", "February 2026"]
    assert [d[1] for d in dense] == [20, 0, 0, 80]


def test_densify_is_a_noop_for_already_contiguous_months():
    counts = [("January 2026", 5), ("February 2026", 7)]
    assert vc.densify_monthly_counts(counts) == counts


# ── Aggregation completeness signalling ──

def test_aggregation_reports_completeness():
    class _Idx:
        def __init__(self, n):
            self.n = n

        def query(self, vector, top_k=10, include_metadata=True, filter=None):
            return {"matches": [{"metadata": {"crop": "Wheat"}} for _ in range(min(self.n, top_k))]}

    # Pass an explicit small page so the test doesn't have to fabricate
    # AGGREGATION_PAGE_SIZE records to exercise the truncation branch.
    partial, complete = vc.fetch_matches_for_aggregation(_Idx(500), None, top_k=100)
    assert complete is False, "a full page means the result set was truncated"

    small, complete2 = vc.fetch_matches_for_aggregation(_Idx(3), None, top_k=100)
    assert complete2 is True


def test_aggregation_page_size_stays_high_enough_to_be_representative():
    # Regression guard: lowering this to 1000 caused live rankings to
    # report "no crop tags were found" because a zero vector gives no
    # similarity ordering, so a smaller top_k returns an arbitrary subset.
    assert vc.AGGREGATION_PAGE_SIZE >= 10000


def test_aggregation_errors_are_not_swallowed_into_empty_results():
    class _Boom:
        def query(self, **kw):
            raise RuntimeError("index unavailable")

    # Returning ([], False) here would be indistinguishable from "the
    # dataset is genuinely empty", which is the worst way for a counting
    # path to fail.
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        vc.fetch_matches_for_aggregation(_Boom(), None)


def test_stats_record_is_excluded_from_aggregation():
    class _Idx:
        def query(self, vector, top_k=10, include_metadata=True, filter=None):
            return {"matches": [
                {"metadata": {"is_stats_record": True, "max_year": "2026"}},
                {"metadata": {"crop": "Wheat"}},
            ]}

    matches, _ = vc.fetch_matches_for_aggregation(_Idx(), None)
    assert len(matches) == 1
    assert matches[0]["metadata"]["crop"] == "Wheat"


def test_ranking_says_lower_bound_when_the_fetch_was_capped(monkeypatch, fake_pinecone_factory):
    # Shrink the page rather than fabricating 10k records.
    monkeypatch.setattr(vc, "AGGREGATION_PAGE_SIZE", 20)
    records = [
        make_record("January", "2026", "negative", "Complaint/Negative Feedback", f"x{i}", crop="Wheat")
        for i in range(50)
    ]
    fake_pinecone_factory(records)
    state = vc.process_chat_query("which crop generated the highest number of complaints?", "fake-key")
    assert "lower bounds" in state["reply"], \
        "a truncated fetch must not be presented as an exhaustive census"
