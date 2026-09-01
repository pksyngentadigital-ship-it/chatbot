"""
Regression tests for relative-date parsing (Phase 3) and the "the requested
period" copy fix (Phase 2) — the correctness issues identified during the
live QA pass.
"""
from conftest import FakeIndex, make_record
import vog_core as vc


# ── month index arithmetic ──

def test_month_idx_round_trip():
    for month, year in [("January", "2026"), ("December", "2025"), ("June", "2024")]:
        idx = vc._month_idx(month, year)
        assert vc._idx_to_month_year(idx) == (month, year)


def test_month_idx_ordering_across_year_boundary():
    dec_idx = vc._month_idx("December", "2025")
    jan_idx = vc._month_idx("January", "2026")
    assert jan_idx == dec_idx + 1


# ── detect_relative_window ──

def test_detect_relative_window_last_n_months():
    assert vc.detect_relative_window("show complaints for the last 3 months") == {"kind": "last_n_months", "n": 3}


def test_detect_relative_window_last_n_days_approximates_to_months():
    assert vc.detect_relative_window("what happened in the last 30 days?") == {"kind": "last_n_months", "n": 1}
    assert vc.detect_relative_window("complaints from the last 90 days") == {"kind": "last_n_months", "n": 3}


def test_detect_relative_window_last_month():
    assert vc.detect_relative_window("what were complaints last month?") == {"kind": "last_n_months", "n": 1}


def test_detect_relative_window_this_month():
    assert vc.detect_relative_window("show sentiment for this month") == {"kind": "this_month"}


def test_detect_relative_window_last_quarter():
    assert vc.detect_relative_window("summarize last quarter") == {"kind": "last_quarter"}


def test_detect_relative_window_this_quarter():
    assert vc.detect_relative_window("what happened this quarter") == {"kind": "this_quarter"}


def test_detect_relative_window_since_month():
    assert vc.detect_relative_window("show feedback since march") == {"kind": "since_month", "month": "March"}


def test_detect_relative_window_none_for_explicit_month():
    assert vc.detect_relative_window("complaints in january 2026") is None


def test_detect_relative_window_none_for_unrelated_query():
    assert vc.detect_relative_window("what do growers think about isabion?") is None


# ── resolve_relative_window ──

def _index_with_months(*month_years):
    records = [make_record(m, y, "neutral", "Others", "x") for m, y in month_years]
    return FakeIndex(records)


def test_resolve_last_n_months_anchors_to_latest_data_not_calendar():
    index = _index_with_months(("January", "2026"), ("February", "2026"), ("March", "2026"))
    months, label = vc.resolve_relative_window({"kind": "last_n_months", "n": 2}, index)
    assert months == [("February", "2026"), ("March", "2026")]
    assert "2" in label


def test_resolve_this_month_is_latest_month_in_data():
    index = _index_with_months(("January", "2026"), ("March", "2026"))
    months, label = vc.resolve_relative_window({"kind": "this_month"}, index)
    assert months == [("March", "2026")]


def test_resolve_last_quarter_spans_correct_calendar_months():
    # Latest data is March 2026 (Q1 = Jan-Mar) -> last quarter is Q4 2025 (Oct-Dec).
    index = _index_with_months(("March", "2026"))
    months, label = vc.resolve_relative_window({"kind": "last_quarter"}, index)
    assert months == [("October", "2025"), ("November", "2025"), ("December", "2025")]


def test_resolve_this_quarter_only_includes_months_up_to_latest():
    # Latest data is February 2026 (Q1 runs Jan-Mar) -> so far only Jan-Feb,
    # not the not-yet-reached March.
    index = _index_with_months(("January", "2026"), ("February", "2026"))
    months, label = vc.resolve_relative_window({"kind": "this_quarter"}, index)
    assert months == [("January", "2026"), ("February", "2026")]


def test_resolve_since_month_uses_previous_year_if_month_not_yet_reached():
    # Latest data is February 2026; "since March" hasn't happened yet this
    # year, so it should mean March 2025 through February 2026.
    index = _index_with_months(("February", "2026"))
    months, label = vc.resolve_relative_window({"kind": "since_month", "month": "March"}, index)
    assert months[0] == ("March", "2025")
    assert months[-1] == ("February", "2026")


def test_resolve_relative_window_none_when_index_has_no_dated_records():
    assert vc.resolve_relative_window({"kind": "this_month"}, FakeIndex([])) is None


# ── DEFAULT_TIMEFRAME_LABEL copy fix ──

def test_default_timeframe_label_is_not_the_old_confusing_string():
    assert vc.DEFAULT_TIMEFRAME_LABEL != "the requested period"


def test_header_uses_new_default_label_when_no_date_resolved():
    header = vc.build_header("sentiment", vc.DEFAULT_TIMEFRAME_LABEL, None, [])
    assert "the requested period" not in header
    assert vc.DEFAULT_TIMEFRAME_LABEL in header
