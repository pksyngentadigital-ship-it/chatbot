"""
Deterministic counting must always be exact Python arithmetic over metadata
tags — never left to the LLM. These tests pin down that contract.
"""
import legacy_api as vc


def test_rank_by_field_counts_comma_separated_tags():
    matches = [
        {"metadata": {"products": "Isabion,Axial"}},
        {"metadata": {"products": "Isabion"}},
        {"metadata": {"products": "Axial"}},
        {"metadata": {"products": ""}},
        {"metadata": {}},
    ]
    ranking = vc.rank_by_field(matches, "products", top_n=10)
    assert dict(ranking) == {"Isabion": 2, "Axial": 2}


def test_rank_by_field_respects_top_n():
    matches = [{"metadata": {"crop": f"Crop{i}"}} for i in range(5)]
    ranking = vc.rank_by_field(matches, "crop", top_n=3)
    assert len(ranking) == 3


def test_rank_by_field_most_common_first():
    matches = (
        [{"metadata": {"crop": "Wheat"}}] * 5
        + [{"metadata": {"crop": "Cotton"}}] * 2
    )
    ranking = vc.rank_by_field(matches, "crop", top_n=10)
    assert ranking[0] == ("Wheat", 5)
    assert ranking[1] == ("Cotton", 2)


def test_compute_monthly_trend_sorted_chronologically_across_years():
    matches = [
        {"metadata": {"month": "March", "year": "2026"}},
        {"metadata": {"month": "January", "year": "2026"}},
        {"metadata": {"month": "December", "year": "2025"}},
        {"metadata": {"month": "January", "year": "2026"}},
    ]
    trend = vc.compute_monthly_trend(matches)
    assert trend == [
        ("December 2025", 1),
        ("January 2026", 2),
        ("March 2026", 1),
    ]


def test_compute_monthly_trend_ignores_records_missing_month_or_year():
    matches = [
        {"metadata": {"month": "January", "year": "2026"}},
        {"metadata": {"month": None, "year": "2026"}},
        {"metadata": {"month": "January", "year": None}},
        {"metadata": {"month": "NotAMonth", "year": "2026"}},
    ]
    trend = vc.compute_monthly_trend(matches)
    assert trend == [("January 2026", 1)]


def test_compute_growth_series_first_period_is_none():
    growth = vc.compute_growth_series([("Jan 2026", 10), ("Feb 2026", 15)])
    assert growth[0] is None


def test_compute_growth_series_percentage_is_correct():
    growth = vc.compute_growth_series([("Jan 2026", 10), ("Feb 2026", 15)])
    assert growth[1] == 50.0


def test_compute_growth_series_handles_zero_previous_count():
    growth = vc.compute_growth_series([("Jan 2026", 0), ("Feb 2026", 5)])
    assert growth[1] is None


def test_compute_growth_series_negative_growth():
    growth = vc.compute_growth_series([("Jan 2026", 20), ("Feb 2026", 10)])
    assert growth[1] == -50.0
