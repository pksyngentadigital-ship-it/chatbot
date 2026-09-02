"""
Ingestion correctness tests.

Every case here corresponds to a defect found by executing the real
parsers against synthetic workbooks. The pre-existing ingestion tests all
passed while these bugs were live, which is precisely why they are worth
pinning down: the failure mode is silent — wrong metadata is written, the
call reports success, and nothing surfaces until an answer looks odd.
"""
from io import BytesIO

import pandas as pd
import pytest

import legacy_api as vc
from vog import retrieval


class _RecordingIndex:
    def __init__(self):
        self.records = {}
        self.delete_all_called = False

    def upsert(self, vectors):
        for v in vectors:
            self.records[v["id"]] = v["metadata"]

    def delete(self, delete_all=False, **kw):
        if delete_all:
            self.delete_all_called = True
            self.records.clear()

    def query(self, vector, top_k=10, include_metadata=True, filter=None):
        return {"matches": [{"metadata": m} for m in list(self.records.values())[:top_k]]}


class _FakePC:
    class _Inf:
        def embed(self, model, inputs, parameters):
            dim = parameters.get("dimension", vc.EMBEDDING_DIMENSION)
            return [type("V", (), {"values": [0.0] * dim})() for _ in inputs]

    def __init__(self, index):
        self.inference = self._Inf()
        self._index = index

    def Index(self, name):
        return self._index


@pytest.fixture
def ingest(monkeypatch):
    """Returns run(sheets_dict, **kw) -> (result, index)."""
    def _run(sheets: dict, **kwargs):
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name, index=False, header=kwargs.pop("_header", True))
        index = _RecordingIndex()
        monkeypatch.setattr(retrieval, "connect", lambda api_key: (_FakePC(index), index))
        result = vc.run_ingestion(buf.getvalue(), "fake-key", **kwargs)
        return result, index
    return _run


def _metas(index):
    """Feedback records only — the index also holds a single stats record
    describing the dataset's extents, which retrieval excludes by the same
    is_stats_record flag."""
    return [m for m in index.records.values() if not m.get("is_stats_record")]


# ── Vector identity ──

def test_no_id_collision_between_row_and_bullet_indexes(ingest):
    # Row 1 with 13 bullets and row 11 with 3 previously produced
    # overlapping ids ("...112"), silently destroying the earlier record.
    many = "\n".join(f"Point number {i} about pricing" for i in range(13))
    few = "\n".join(f"Row eleven point {i}" for i in range(3))
    rows = ["Complaint/Negative Feedback"] * 12
    cells = [""] * 12
    rows[1], cells[1] = "Complaint/Negative Feedback", many
    rows[11], cells[11] = "Complaint/Negative Feedback", few
    df = pd.DataFrame({"Category": rows, "1st Week January": cells})

    result, index = ingest({"2026": df})
    assert result["total_records"] == 16
    assert len(_metas(index)) == 16, "reported count must equal what actually landed"


def test_reingesting_identical_content_is_idempotent(ingest):
    df = pd.DataFrame({
        "Category": ["Positive Feedback"],
        "1st Week January": ["Isabion worked well\nAxial also good"],
    })
    r1, idx1 = ingest({"2026": df})
    r2, idx2 = ingest({"2026": df})
    assert sorted(idx1.records.keys()) == sorted(idx2.records.keys()), \
        "content-hash ids must be stable across runs"


# ── Layout A ──

def test_merged_category_cells_are_forward_filled(ingest):
    # Merged cells give the value only to the first row; continuation rows
    # came back NaN and every one of them was silently dropped.
    df = pd.DataFrame({
        "Category": ["Complaint/Negative Feedback", None, None],
        "1st Week January": ["First complaint here", "Second complaint here", "Third complaint here"],
    })
    result, index = ingest({"2026": df})
    assert result["total_records"] == 3, "continuation rows of a merged category must not be dropped"


def test_category_spelling_variants_keep_their_sentiment(ingest):
    df = pd.DataFrame({
        "Category": ["Complaint / Negative Feedback", "Positive Feedbacks"],
        "1st Week January": ["A delivery problem", "Really good results"],
    })
    result, index = ingest({"2026": df})
    sentiments = {m["sentiment"] for m in _metas(index)}
    assert sentiments == {"negative", "positive"}, \
        "spacing/plural variants must not degrade to neutral"


def test_unrecognised_category_is_reported_not_silently_neutralised(ingest):
    df = pd.DataFrame({
        "Category": ["Complaint/Negative Feedback", "Totally Unknown Bucket"],
        "1st Week January": ["A real complaint", "Some other text"],
    })
    result, index = ingest({"2026": df})
    assert result["total_records"] == 1
    reasons = {s["reason"] for s in result["skipped"]}
    assert "unmapped_categories" in reasons


def test_week_column_without_a_month_is_skipped_and_reported(ingest):
    # A second, valid sheet means ingestion succeeds overall — the point is
    # that the broken sheet is REPORTED rather than silently contributing
    # nothing while the call still claims success.
    good = pd.DataFrame({"Category": ["Positive Feedback"], "1st Week January": ["A good result here"]})
    bad = pd.DataFrame({"Category": ["Positive Feedback"], "Week 1": ["Unreachable feedback here"]})
    result, index = ingest({"VOG 2026": good, "VOG 2025": bad})
    assert any(s["reason"] == "unparseable_month" for s in result["skipped"])
    assert result["total_records"] == 1


def test_non_month_column_headers_do_not_become_months(ingest):
    # 'Remarks' used to parse as March via 're-MAR-ks'.
    for header in ["Remarks", "Summary", "Decrease"]:
        assert vc.extract_month_from_col(header) == "Unknown", header


# ── Layout B ──

def _layout_b(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def test_layout_b_only_ingests_known_category_columns(ingest):
    # Region / Remarks were being ingested as feedback, and region names
    # were then counted as products.
    df = _layout_b(
        [["January", "1st", "Maharashtra", "Great results with Isabion", "Follow up with dealer"]],
        ["Month", "Week", "Region", "Positive Feedback", "Remarks"],
    )
    result, index = ingest({"Legacy 2025": df})
    cats = {m["category"] for m in _metas(index)}
    assert cats == {"Positive Feedback"}
    assert not any("Maharashtra" in m["value"] for m in _metas(index))
    assert any(s["reason"] == "ignored_columns" for s in result["skipped"])


def test_layout_b_week_resets_when_the_month_changes(ingest):
    # February's first row left Week blank and inherited January's "5th".
    df = _layout_b(
        [["January", "1st", "Jan first block"],
         ["", "5th", "Jan fifth week"],
         ["February", "", "Feb first block feedback"]],
        ["Month", "Week", "Positive Feedback"],
    )
    result, index = ingest({"Legacy 2025": df})
    feb = [m for m in _metas(index) if m["month"] == "February"]
    assert feb, "February row should be ingested"
    assert "5th" not in feb[0]["week"], "stale week must not bleed across a month boundary"


def test_layout_b_accepts_week_column_aliases(ingest):
    df = _layout_b(
        [["January", "1st", "Axial delivery was delayed"]],
        ["Month", "Week No", "Complaint/Negative Feedback"],
    )
    result, index = ingest({"Legacy 2025": df})
    metas = _metas(index)
    assert metas and metas[0]["week_num"] == 1, "'Week No' must be recognised as the week column"


def test_layout_b_total_row_does_not_overwrite_the_month(ingest):
    df = _layout_b(
        [["January", "1st", "January complaint about Axial"],
         ["Total", "", "Total row junk text here"],
         ["February", "1st", "February complaint about Tilt"]],
        ["Month", "Week", "Complaint/Negative Feedback"],
    )
    result, index = ingest({"Legacy 2025": df})
    months = {m["month"] for m in _metas(index)}
    assert "Unknown" not in months, "an unparseable month cell must not poison the forward-fill"


# ── Cell hygiene ──

def test_date_and_numeric_cells_are_not_ingested_as_feedback(ingest):
    df = pd.DataFrame({
        "Category": ["Others", "Others"],
        "1st Week January": [pd.Timestamp("2026-01-05"), "A genuine piece of feedback"],
    })
    result, index = ingest({"2026": df})
    values = [m["value"] for m in _metas(index)]
    assert values == ["A genuine piece of feedback"]


def test_inline_bullet_glyphs_split_into_separate_records(ingest):
    df = pd.DataFrame({
        "Category": ["Complaint/Negative Feedback"],
        "1st Week January": ["• Price is high • Product unavailable • App is slow"],
    })
    result, index = ingest({"2026": df})
    assert result["total_records"] == 3, "inline bullets must not merge into one record"


def test_duplicate_columns_are_rejected_not_ingested_as_pandas_repr(ingest):
    # "1st Week January" and "1st  Week  January" are distinct raw headers
    # that collapse to the same label once whitespace is normalized. row[col]
    # then returns a Series whose repr ('Name: 0, dtype: str') was being
    # embedded as feedback.
    good = pd.DataFrame({"Category": ["Positive Feedback"], "1st Week January": ["A good result here"]})
    dupe = pd.DataFrame({
        "Category": ["Positive Feedback"],
        "1st Week January": ["aaa good result"],
        "1st  Week  January": ["bbb also good"],
    })
    result, index = ingest({"VOG 2026": good, "VOG 2025": dupe})
    assert any(s["reason"] == "duplicate_columns" for s in result["skipped"])
    assert not any("dtype" in m["value"] for m in _metas(index))


# ── Provenance & purge ──

def test_provenance_metadata_is_written(ingest):
    df = pd.DataFrame({"Category": ["Positive Feedback"], "1st Week January": ["Good stuff here"]})
    result, index = ingest({"2026": df})
    m = _metas(index)[0]
    assert m["sheet"] == "2026"
    assert m["src_row"] >= 0
    assert m["ingest_run"] == result["ingest_run"]


def test_purge_first_clears_the_index_before_writing(ingest):
    df = pd.DataFrame({"Category": ["Positive Feedback"], "1st Week January": ["Good stuff here"]})
    result, index = ingest({"2026": df}, purge_first=True)
    assert index.delete_all_called is True
    assert len(_metas(index)) == result["total_records"]


def test_oversized_cell_is_truncated_not_left_to_blow_the_metadata_limit(ingest):
    huge = "x" * 30000
    df = pd.DataFrame({"Category": ["Others"], "1st Week January": [huge]})
    result, index = ingest({"2026": df})
    m = _metas(index)[0]
    assert len(m["value"]) <= vc.MAX_VALUE_CHARS
    assert m.get("truncated") is True


def test_text_field_no_longer_duplicates_value(ingest):
    df = pd.DataFrame({"Category": ["Others"], "1st Week January": ["Some feedback text"]})
    result, index = ingest({"2026": df})
    assert "text" not in _metas(index)[0]


# ── Year inference ──

def test_year_inference_uses_the_nearest_dated_sheet():
    sheets = ["Legacy A", "VOG 2024", "VOG 2026"]
    # Nearest dated neighbour to index 0 is VOG 2024 (later) -> 2023.
    assert vc.infer_year_for_sheet("Legacy A", sheets) == "2023"


def test_year_inference_refuses_to_duplicate_an_existing_year():
    sheets = ["Legacy Jan-Jun", "VOG 2026", "VOG 2025"]
    # The naive guess (2025) collides with a real dated sheet, which would
    # double-count every record; refusing is correct.
    assert vc.infer_year_for_sheet("Legacy Jan-Jun", sheets) is None


def test_dry_run_parses_fully_without_a_key_or_a_write(monkeypatch):
    """The CLI's --dry-run must be the real parse result, so it can vet a
    workbook against production's rules without touching production."""
    def _boom(api_key):
        raise AssertionError("dry_run must not open a Pinecone connection")
    monkeypatch.setattr(retrieval, "connect", _boom)

    df = pd.DataFrame({
        "Category": ["Positive Feedback", "Not A Real Category"],
        "1st Week January": ["Isabion did well on wheat", "should be skipped"],
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="2026", index=False)

    result = vc.run_ingestion(buf.getvalue(), None, dry_run=True)
    assert result["dry_run"] is True
    assert result["total_records"] == 1
    assert result["summary"] == {"January 2026": 1}
    assert [s["reason"] for s in result["skipped"]] == ["unmapped_categories"]


def test_a_write_without_a_key_is_refused_rather_than_half_done():
    df = pd.DataFrame({"Category": ["Positive Feedback"],
                       "1st Week January": ["Isabion did well"]})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="2026", index=False)

    with pytest.raises(ValueError, match="API key is required"):
        vc.run_ingestion(buf.getvalue(), None)
