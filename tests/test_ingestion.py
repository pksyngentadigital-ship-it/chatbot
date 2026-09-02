"""
Excel ingestion parser — must handle both real sheet layouts and correctly
tag month/year/sentiment/category/crop/product metadata on each row.
"""
from io import BytesIO

import pandas as pd
import pytest

import legacy_api as vc
from vog import retrieval


def _build_workbook(sheets: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def test_ingestion_layout_a_categories_as_rows_weeks_as_columns(fake_pinecone_factory):
    df = pd.DataFrame({
        "Category": ["Positive Feedback", "Complaint/Negative Feedback"],
        "1st Week January": [
            "Growers were happy with Isabion results on wheat.",
            "",
        ],
        "2nd Week January": [
            "",
            "Delayed delivery of Axial reported by growers.",
        ],
    })
    workbook = _build_workbook({"2026": df})
    client = fake_pinecone_factory([])

    result = vc.run_ingestion(workbook, "fake-key")

    assert result["total_records"] == 2
    metas = [r["metadata"] for r in client.Index("chatbot").records
             if not r["metadata"].get("is_stats_record")]
    sentiments = {m["sentiment"] for m in metas}
    assert sentiments == {"positive", "negative"}
    assert all(m["year"] == "2026" for m in metas)
    assert all(m["month"] == "January" for m in metas)
    isabion_row = next(m for m in metas if m["sentiment"] == "positive")
    assert "Isabion" in isabion_row["products"]
    assert "Wheat" in isabion_row["crop"]


def test_ingestion_layout_b_month_week_rows_category_columns(fake_pinecone_factory):
    df = pd.DataFrame({
        "Month": ["January", "February"],
        "Week": ["1st", "1st"],
        "Positive Feedback": [
            "Growers appreciated Tilt performance on cotton.",
            "",
        ],
        "Complaint/Negative Feedback": [
            "",
            "Pricing concerns raised about Score.",
        ],
    })
    workbook = _build_workbook({"Legacy 2025": df})
    client = fake_pinecone_factory([])

    result = vc.run_ingestion(workbook, "fake-key")

    assert result["total_records"] == 2
    metas = [r["metadata"] for r in client.Index("chatbot").records
             if not r["metadata"].get("is_stats_record")]
    assert {m["month"] for m in metas} == {"January", "February"}
    assert all(m["year"] == "2025" for m in metas)


def test_ingestion_raises_when_workbook_has_no_recognizable_data(fake_pinecone_factory):
    df = pd.DataFrame({"Unrelated Column": ["nothing useful here"]})
    workbook = _build_workbook({"2026": df})
    fake_pinecone_factory([])

    with pytest.raises(ValueError):
        vc.run_ingestion(workbook, "fake-key")


def test_ingestion_skips_cells_that_are_only_empty_markers(fake_pinecone_factory):
    # A sheet where every cell is an "empty" marker (N/A) yields zero
    # records — same as an unrecognized layout, so it raises ValueError
    # rather than silently ingesting nothing.
    df = pd.DataFrame({
        "Category": ["Positive Feedback"],
        "1st Week January": ["N/A"],
    })
    workbook = _build_workbook({"2026": df})
    fake_pinecone_factory([])

    with pytest.raises(ValueError):
        vc.run_ingestion(workbook, "fake-key")
