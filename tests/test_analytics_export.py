"""
tests/test_analytics_export.py — M5 dashboard statistics, filters and exports.

WHAT IS TESTED
──────────────
  core.analytics
    - Sentiment distribution counts and percentages (successful rows only).
    - Category distribution counts (all seven categories, zero-filled).
    - Skipped / failed rows excluded from sentiment & category stats.
    - Behaviour when there are no successful rows.
    - filter_results() narrows the view without mutating the source.

  core.exporter
    - CSV export keeps every original + ai_* column, UTF-8-SIG, Indonesian text.
    - Excel export is a valid workbook readable by pandas, sheet "Hasil Analisis".
    - Both exports keep succeeded, skipped and failed rows.

  UI (streamlit AppTest)
    - Dashboard + table + export render from session data without calling the AI.
    - Dashboard shows an informative message, not charts, when nothing succeeded.

All data is dummy data built via run_batch() with analyze_feedback mocked.

HOW TO RUN:
    pytest tests/test_analytics_export.py -v
"""

from __future__ import annotations

import io
from unittest.mock import patch

import openpyxl
import pandas as pd
import pytest

from core.analytics import (
    CATEGORY_ORDER,
    SENTIMENT_ORDER,
    compute_dashboard_stats,
    filter_results,
)
from core.batch_analyzer import RowStatus, results_to_dataframe, run_batch
from core.exporter import EXCEL_SHEET_NAME, to_csv_bytes, to_excel_bytes
from llm.schemas import Category, FeedbackAnalysis, Sentiment

AI_COLUMNS = [
    "ai_original_text",
    "ai_sentiment",
    "ai_category",
    "ai_key_issue",
    "ai_evidence",
    "ai_recommendation",
    "ai_status",
    "ai_error",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _analysis(text: str, sentiment: Sentiment, category: Category) -> FeedbackAnalysis:
    return FeedbackAnalysis(
        original_text=text,
        sentiment=sentiment,
        category=category,
        key_issue="Masalah utama pelanggan",
        evidence=text,
        recommendation="Tingkatkan kualitas layanan",
    )


# review text → (sentiment, category); "GAGAL" rows raise; blank rows are skipped
_PLAN = {
    "Pengiriman sangat cepat, mantap!": (Sentiment.POSITIVE, Category.DELIVERY),
    "Kurir terlambat tiga hari": (Sentiment.NEGATIVE, Category.DELIVERY),
    "Harganya terlalu mahal": (Sentiment.NEGATIVE, Category.PRICE),
    "Aplikasinya biasa saja": (Sentiment.NEUTRAL, Category.APP_USABILITY),
}


def _fake_analyze(text: str) -> FeedbackAnalysis:
    if text.startswith("GAGAL"):
        raise RuntimeError("boom")
    sentiment, category = _PLAN[text]
    return _analysis(text, sentiment, category)


@pytest.fixture
def original_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6", "7"],
            "ulasan": [
                "Pengiriman sangat cepat, mantap!",
                "Kurir terlambat tiga hari",
                "",                       # skipped
                "Harganya terlalu mahal",
                "GAGAL diproses",         # failed
                "Aplikasinya biasa saja",
                None,                     # skipped
            ],
            "kota": ["Jakarta", "Bandung", "Surabaya", "Medan", "Bogor", "Depok", "Bekasi"],
        }
    )


@pytest.fixture
def merged_df(original_df: pd.DataFrame) -> pd.DataFrame:
    with patch("core.batch_analyzer.analyze_feedback", side_effect=_fake_analyze):
        result = run_batch(original_df, "ulasan", batch_delay_seconds=0)
    return results_to_dataframe(original_df, result)


@pytest.fixture
def no_success_df(original_df: pd.DataFrame) -> pd.DataFrame:
    with patch("core.batch_analyzer.analyze_feedback", side_effect=RuntimeError("down")):
        result = run_batch(original_df, "ulasan", batch_delay_seconds=0)
    return results_to_dataframe(original_df, result)


# ── Dashboard statistics ──────────────────────────────────────────────────────

class TestDashboardStats:
    def test_status_counts(self, merged_df):
        stats = compute_dashboard_stats(merged_df)
        assert (stats.total, stats.succeeded, stats.skipped, stats.failed) == (7, 4, 2, 1)

    def test_sentiment_distribution_counts(self, merged_df):
        counts = compute_dashboard_stats(merged_df).sentiment_counts
        assert counts["label"].tolist() == SENTIMENT_ORDER
        assert dict(zip(counts["label"], counts["count"])) == {
            "Positive": 1, "Negative": 2, "Neutral": 1,
        }

    def test_sentiment_percentages(self, merged_df):
        counts = compute_dashboard_stats(merged_df).sentiment_counts
        pct = dict(zip(counts["label"], counts["percent"]))
        assert pct == {"Positive": 25.0, "Negative": 50.0, "Neutral": 25.0}
        assert sum(pct.values()) == pytest.approx(100.0)

    def test_category_distribution_counts(self, merged_df):
        counts = compute_dashboard_stats(merged_df).category_counts
        assert counts["label"].tolist() == CATEGORY_ORDER
        by_label = dict(zip(counts["label"], counts["count"]))
        assert by_label["Delivery"] == 2
        assert by_label["Price"] == 1
        assert by_label["Application / Usability"] == 1
        for zero in ("Product Quality", "Customer Service", "Payment", "Other"):
            assert by_label[zero] == 0

    def test_skipped_and_failed_excluded_from_distributions(self, merged_df):
        stats = compute_dashboard_stats(merged_df)
        assert stats.sentiment_counts["count"].sum() == stats.succeeded == 4
        assert stats.category_counts["count"].sum() == stats.succeeded

    def test_stray_values_on_non_success_rows_are_ignored(self, merged_df):
        # Even if a failed row somehow carried a sentiment, it must not be counted.
        tampered = merged_df.copy()
        failed_idx = tampered.index[tampered["ai_status"] == RowStatus.FAILED.value]
        tampered.loc[failed_idx, "ai_sentiment"] = "Positive"
        tampered.loc[failed_idx, "ai_category"] = "Payment"
        stats = compute_dashboard_stats(tampered)
        counts = dict(zip(stats.sentiment_counts["label"], stats.sentiment_counts["count"]))
        assert counts["Positive"] == 1
        cats = dict(zip(stats.category_counts["label"], stats.category_counts["count"]))
        assert cats["Payment"] == 0

    def test_no_success_results(self, no_success_df):
        stats = compute_dashboard_stats(no_success_df)
        assert not stats.has_success
        assert (stats.total, stats.succeeded, stats.skipped, stats.failed) == (7, 0, 2, 5)
        assert stats.sentiment_counts["count"].sum() == 0
        assert stats.sentiment_counts["percent"].sum() == 0
        assert stats.category_counts["count"].sum() == 0


# ── Filters ───────────────────────────────────────────────────────────────────

class TestFilterResults:
    def test_no_filter_returns_all_rows(self, merged_df):
        assert len(filter_results(merged_df)) == len(merged_df)

    def test_filter_by_sentiment(self, merged_df):
        view = filter_results(merged_df, sentiments=["Negative"])
        assert len(view) == 2
        assert set(view["ai_sentiment"]) == {"Negative"}

    def test_filter_by_category(self, merged_df):
        view = filter_results(merged_df, categories=["Delivery"])
        assert view["id"].tolist() == ["1", "2"]

    def test_combined_filters(self, merged_df):
        view = filter_results(merged_df, sentiments=["Negative"], categories=["Delivery"])
        assert view["id"].tolist() == ["2"]

    def test_filter_does_not_mutate_source_or_stats(self, merged_df):
        before = merged_df.copy()
        stats_before = compute_dashboard_stats(merged_df)
        filter_results(merged_df, sentiments=["Positive"], categories=["Delivery"])
        pd.testing.assert_frame_equal(merged_df, before)
        assert compute_dashboard_stats(merged_df).succeeded == stats_before.succeeded


# ── CSV export ────────────────────────────────────────────────────────────────

class TestCsvExport:
    def test_has_utf8_bom(self, merged_df):
        assert to_csv_bytes(merged_df).startswith(b"\xef\xbb\xbf")

    def test_all_columns_present(self, merged_df, original_df):
        back = pd.read_csv(io.BytesIO(to_csv_bytes(merged_df)), encoding="utf-8-sig", dtype=str)
        assert list(back.columns) == list(original_df.columns) + AI_COLUMNS

    def test_preserves_indonesian_text(self):
        df = pd.DataFrame({
            "ulasan": ["Pelayanan ramah sekali, terima kasih! Harga “terjangkau” — mantap 👍"],
            "ai_key_issue": ["Pelanggan puas dengan keramahan & harga"],
        })
        back = pd.read_csv(io.BytesIO(to_csv_bytes(df)), encoding="utf-8-sig")
        assert back.loc[0, "ulasan"] == df.loc[0, "ulasan"]
        assert back.loc[0, "ai_key_issue"] == df.loc[0, "ai_key_issue"]

    def test_keeps_all_statuses(self, merged_df):
        back = pd.read_csv(io.BytesIO(to_csv_bytes(merged_df)), encoding="utf-8-sig", dtype=str)
        assert len(back) == 7
        assert back["ai_status"].value_counts().to_dict() == {
            RowStatus.SUCCESS.value: 4, RowStatus.SKIPPED.value: 2, RowStatus.FAILED.value: 1,
        }
        assert back.loc[back["ai_status"] == RowStatus.FAILED.value, "ai_error"].notna().all()


# ── Excel export ──────────────────────────────────────────────────────────────

class TestExcelExport:
    def test_readable_by_pandas_with_all_columns(self, merged_df, original_df):
        back = pd.read_excel(io.BytesIO(to_excel_bytes(merged_df)), sheet_name=EXCEL_SHEET_NAME, dtype=str)
        assert list(back.columns) == list(original_df.columns) + AI_COLUMNS
        assert len(back) == len(merged_df)

    def test_sheet_name(self, merged_df):
        wb = openpyxl.load_workbook(io.BytesIO(to_excel_bytes(merged_df)))
        assert wb.sheetnames == ["Hasil Analisis"]

    def test_keeps_all_statuses(self, merged_df):
        back = pd.read_excel(io.BytesIO(to_excel_bytes(merged_df)), dtype=str)
        assert back["ai_status"].value_counts().to_dict() == {
            RowStatus.SUCCESS.value: 4, RowStatus.SKIPPED.value: 2, RowStatus.FAILED.value: 1,
        }

    def test_preserves_indonesian_text(self, merged_df):
        back = pd.read_excel(io.BytesIO(to_excel_bytes(merged_df)), dtype=str)
        assert back.loc[0, "ulasan"] == "Pengiriman sangat cepat, mantap!"
        assert back.loc[0, "ai_recommendation"] == "Tingkatkan kualitas layanan"

    def test_formula_like_text_stored_as_text(self):
        df = pd.DataFrame({"ulasan": ["=1+1 bagus", "=HYPERLINK(\"x\")"]})
        wb = openpyxl.load_workbook(io.BytesIO(to_excel_bytes(df)))
        cells = [c for c in wb["Hasil Analisis"]["A"][1:]]
        assert [c.value for c in cells] == df["ulasan"].tolist()
        assert all(c.data_type == "s" for c in cells)


# ── UI integration (no AI calls) ──────────────────────────────────────────────

_UI_SCRIPT = """
import streamlit as st
from ui.analytics import render_dashboard, render_filtered_table, render_export
df = st.session_state["merged"]
render_dashboard(df)
render_filtered_table(df)
render_export(df)
"""


def _chart_count(at) -> int:
    # Element type name differs across Streamlit versions (1.50 vs 1.67+).
    return len(at.get("arrow_vega_lite_chart")) + len(at.get("vega_lite_chart"))


def _run_ui(df: pd.DataFrame):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_UI_SCRIPT, default_timeout=30)
    at.session_state["merged"] = df
    return at


class TestUiNoApiCalls:
    def test_dashboard_filter_and_export_do_not_call_ai(self, merged_df):
        with patch("core.batch_analyzer.analyze_feedback") as batch_ai, \
             patch("core.analyzer.analyze_feedback") as core_ai, \
             patch("llm.service._call_api") as llm_ai:
            at = _run_ui(merged_df).run()
            assert not at.exception
            # Interact: change filters → rerun.
            at.multiselect(key="batch_filter_category").select("Delivery").run()
            assert not at.exception
            batch_ai.assert_not_called()
            core_ai.assert_not_called()
            llm_ai.assert_not_called()

        assert _chart_count(at) == 2  # sentiment + category
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Total ulasan"] == "7"
        assert metrics["✅ Berhasil"] == "4"
        assert metrics["⏭️ Dilewati"] == "2"
        assert metrics["❌ Gagal"] == "1"
        assert metrics["Negative"] == "50.0%"
        # Filtered table shows only the 2 Delivery rows; source untouched.
        assert len(at.dataframe[0].value) == 2
        assert len(at.session_state["merged"]) == 7
        # Export bytes generated and cached from session data.
        cache = at.session_state["batch_export_cache"]
        assert cache["csv"].startswith(b"\xef\xbb\xbf")
        assert cache["xlsx"][:2] == b"PK"

    def test_dashboard_without_success_shows_message(self, no_success_df):
        at = _run_ui(no_success_df).run()
        assert not at.exception
        assert any("Belum ada ulasan yang berhasil" in i.value for i in at.info)
        # No chart elements rendered when there is nothing to plot.
        assert _chart_count(at) == 0


# ── End-to-end batch tab flow (upload → run → dashboard → new upload) ────────

_BATCH_TAB_SCRIPT = """
import io
from unittest.mock import patch
import streamlit as st
import tests.test_analytics_export as t
from ui.batch_input import render_batch_input_section

class _Upload(io.BytesIO):
    name = "feedback.csv"

payload = st.session_state.get("csv_payload")
calls = st.session_state.setdefault("ai_calls", [])

def _fake(text):
    calls.append(text)
    return t._fake_analyze(text)

with patch("streamlit.file_uploader", return_value=_Upload(payload) if payload else None), \\
     patch("core.batch_analyzer.analyze_feedback", side_effect=_fake):
    render_batch_input_section()
"""

_CSV_A = (
    'id,ulasan\n1,"Pengiriman sangat cepat, mantap!"\n2,Kurir terlambat tiga hari\n'
    "3,\n4,GAGAL diproses\n"
).encode("utf-8")
_CSV_B = "id,ulasan\nA,Harganya terlalu mahal\nB,Aplikasinya biasa saja\n".encode("utf-8")


class TestBatchTabFlow:
    def _metrics(self, at):
        return {m.label: m.value for m in at.metric}

    def test_dashboard_updates_on_new_upload_without_extra_ai_calls(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(_BATCH_TAB_SCRIPT, default_timeout=30)
        at.session_state["csv_payload"] = _CSV_A
        at.run()
        assert not at.exception
        assert at.session_state["ai_calls"] == []          # nothing before clicking

        at.button(key="batch_run_button").click().run()
        assert not at.exception
        assert len(at.session_state["ai_calls"]) == 3       # 3 non-blank rows
        m = self._metrics(at)
        assert (m["Total ulasan"], m["✅ Berhasil"], m["⏭️ Dilewati"], m["❌ Gagal"]) == ("4", "2", "1", "1")
        assert m["Positive"] == "50.0%" and m["Negative"] == "50.0%"

        # Filter + rerun: no additional AI calls, stats unchanged.
        at.multiselect(key="batch_filter_category").select("Delivery").run()
        assert len(at.session_state["ai_calls"]) == 3
        assert self._metrics(at)["✅ Berhasil"] == "2"

        # Upload a different CSV → previous results cleared until a new run.
        at.session_state["csv_payload"] = _CSV_B
        at.run()
        assert not at.exception
        assert "Total ulasan" not in self._metrics(at)
        assert "batch_export_cache" not in at.session_state

        at.button(key="batch_run_button").click().run()
        m = self._metrics(at)
        assert (m["Total ulasan"], m["✅ Berhasil"], m["⏭️ Dilewati"], m["❌ Gagal"]) == ("2", "2", "0", "0")
        assert m["Positive"] == "0.0%" and m["Neutral"] == "50.0%"
        assert len(at.session_state["ai_calls"]) == 5


class TestExcelEngineMissing:
    def test_missing_openpyxl_keeps_csv_and_warns(self, merged_df):
        with patch("ui.analytics.to_excel_bytes", side_effect=ImportError("openpyxl")):
            at = _run_ui(merged_df).run()
        assert not at.exception
        assert any("openpyxl" in w.value for w in at.warning)
        assert at.session_state["batch_export_cache"]["xlsx"] is None
        assert at.session_state["batch_export_cache"]["csv"].startswith(b"\xef\xbb\xbf")
