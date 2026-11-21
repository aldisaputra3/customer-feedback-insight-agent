"""
tests/test_batch_analyzer.py — Unit tests for core/batch_analyzer.py.

WHAT IS TESTED
──────────────
  load_csv():
    - Valid UTF-8 CSV is parsed correctly.
    - Valid CSV with BOM (UTF-8-sig) is parsed correctly.
    - Valid CSV with Latin-1 encoding is parsed correctly.
    - Empty bytes raises CsvValidationError.
    - File exceeding max_size_bytes raises CsvValidationError.
    - File exceeding max_rows raises CsvValidationError.
    - CSV with no data rows raises CsvValidationError.

  run_batch():
    - All rows succeed: BatchResult has correct succeeded count.
    - Blank rows are skipped without calling the API.
    - NaN rows are skipped without calling the API.
    - A failing row is marked FAILED and the next row is still processed.
    - All rows fail: BatchResult has correct failed count, succeeded=0.
    - on_progress callback is called once per row.
    - Returned RowResult.analysis fields match the mock FeedbackAnalysis.
    - Row count in BatchResult equals number of input rows.
    - feedback_column not in DataFrame raises ValueError.

  results_to_dataframe():
    - Merged DataFrame has original columns plus all ai_ columns.
    - SUCCESS rows have populated ai_ analysis columns.
    - SKIPPED rows have empty ai_ analysis columns.
    - FAILED rows have empty ai_ analysis columns and a non-empty ai_error.
    - Row count matches original DataFrame.

No real API key or network access is required.
All analyze_feedback() calls are replaced by unittest.mock patches.

HOW TO RUN:
    pytest tests/test_batch_analyzer.py -v
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.batch_analyzer import (
    BatchResult,
    CsvValidationError,
    RowResult,
    RowStatus,
    load_csv,
    results_to_dataframe,
    run_batch,
)
from llm.schemas import Category, FeedbackAnalysis, Sentiment
from llm.service import (
    AnalysisError,
    AuthenticationError,
    ResponseParseError,
    TransientError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_bytes(content: str, encoding: str = "utf-8") -> bytes:
    return content.encode(encoding)


def _make_analysis(original_text: str = "Produk bagus!") -> FeedbackAnalysis:
    """Return a valid FeedbackAnalysis whose evidence is a substring of original_text."""
    return FeedbackAnalysis(
        original_text=original_text,
        sentiment=Sentiment.POSITIVE,
        category=Category.PRODUCT_QUALITY,
        key_issue="Kualitas produk sangat baik.",
        evidence=original_text.split()[0],
        recommendation="Pertahankan standar kualitas.",
    )


def _simple_df(*texts: str) -> pd.DataFrame:
    """Create a one-column DataFrame with the given review texts."""
    return pd.DataFrame({"review": list(texts)})


# ── load_csv tests ────────────────────────────────────────────────────────────

class TestLoadCsv:
    """CSV parsing and validation."""

    def test_valid_utf8_csv_parsed(self):
        raw = _csv_bytes("review,rating\nProduk bagus,5\nPengiriman lambat,2\n")
        df = load_csv(raw)
        assert list(df.columns) == ["review", "rating"]
        assert len(df) == 2

    def test_utf8_bom_csv_parsed(self):
        """BOM-prefixed UTF-8 (common from Excel) must be handled."""
        raw = "review\nProduk bagus\n".encode("utf-8-sig")
        df = load_csv(raw)
        assert "review" in df.columns

    def test_latin1_csv_parsed(self):
        """Latin-1 encoded CSV (common from Windows) must be handled."""
        # Use only characters in the latin-1 range (avoid Unicode em dash U+2014)
        raw = _csv_bytes("ulasan\nPengiriman lambat - tidak bagus\n", "latin-1")
        df = load_csv(raw)
        assert "ulasan" in df.columns

    def test_empty_bytes_raises(self):
        with pytest.raises(CsvValidationError, match="kosong"):
            load_csv(b"")

    def test_file_exceeds_size_limit_raises(self):
        raw = _csv_bytes("review\n" + "a" * 100)
        with pytest.raises(CsvValidationError, match="terlalu besar"):
            load_csv(raw, max_size_bytes=50)

    def test_file_exceeds_row_limit_raises(self):
        rows = "\n".join(f"review_{i}" for i in range(10))
        raw = _csv_bytes(f"review\n{rows}\n")
        with pytest.raises(CsvValidationError, match="terlalu banyak baris"):
            load_csv(raw, max_rows=3)

    def test_csv_with_no_data_rows_raises(self):
        raw = _csv_bytes("review,rating\n")
        with pytest.raises(CsvValidationError, match="tidak memiliki baris data"):
            load_csv(raw)

    def test_bytesio_source_accepted(self):
        raw = _csv_bytes("review\nBagus sekali\n")
        df = load_csv(io.BytesIO(raw))
        assert len(df) == 1

    def test_all_columns_returned_as_strings(self):
        raw = _csv_bytes("review,rating\nProduk bagus,5\n")
        df = load_csv(raw)
        # dtype=str means numeric columns come back as str
        assert df["rating"].dtype == object


# ── run_batch tests ───────────────────────────────────────────────────────────

class TestRunBatchAllSucceed:
    """All rows produce a valid FeedbackAnalysis."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_all_rows_succeed(self, mock_analyze):
        df = _simple_df("Produk bagus!", "Pengiriman lambat.")
        mock_analyze.side_effect = [
            _make_analysis("Produk bagus!"),
            _make_analysis("Pengiriman lambat."),
        ]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 2
        assert result.skipped  == 0
        assert result.failed   == 0
        assert result.total    == 2

    @patch("core.batch_analyzer.analyze_feedback")
    def test_analysis_fields_stored(self, mock_analyze):
        df = _simple_df("Produk bagus!")
        analysis = _make_analysis("Produk bagus!")
        mock_analyze.return_value = analysis

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.rows[0].status   == RowStatus.SUCCESS
        assert result.rows[0].analysis is analysis

    @patch("core.batch_analyzer.analyze_feedback")
    def test_row_count_equals_input_rows(self, mock_analyze):
        df = _simple_df("A", "B", "C")
        mock_analyze.side_effect = [_make_analysis(t) for t in ["A", "B", "C"]]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert len(result.rows) == 3


class TestRunBatchSkipping:
    """Blank and NaN rows are skipped without API calls."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_blank_rows_skipped(self, mock_analyze):
        df = _simple_df("Produk bagus!", "", "   ")
        mock_analyze.return_value = _make_analysis("Produk bagus!")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 1
        assert result.skipped   == 2
        assert mock_analyze.call_count == 1

    @patch("core.batch_analyzer.analyze_feedback")
    def test_nan_rows_skipped(self, mock_analyze):
        """NaN values (pandas default for missing cells) must be treated as blank."""
        df = pd.DataFrame({"review": ["Produk bagus!", None, float("nan")]})
        mock_analyze.return_value = _make_analysis("Produk bagus!")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 1
        assert result.skipped   == 2
        assert mock_analyze.call_count == 1

    @patch("core.batch_analyzer.analyze_feedback")
    def test_all_blank_rows_no_api_calls(self, mock_analyze):
        df = _simple_df("", "   ", "\n")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        mock_analyze.assert_not_called()
        assert result.skipped == 3
        assert result.succeeded == 0


class TestRunBatchFailures:
    """Failed rows are recorded; processing continues."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_one_failure_rest_processed(self, mock_analyze):
        df = _simple_df("Baris 1", "Baris 2", "Baris 3")
        mock_analyze.side_effect = [
            _make_analysis("Baris 1"),
            TransientError("timeout"),
            _make_analysis("Baris 3"),
        ]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 2
        assert result.failed    == 1
        assert result.skipped   == 0
        assert result.rows[0].status == RowStatus.SUCCESS
        assert result.rows[1].status == RowStatus.FAILED
        assert result.rows[2].status == RowStatus.SUCCESS

    @patch("core.batch_analyzer.analyze_feedback")
    def test_failed_row_stores_error_message(self, mock_analyze):
        df = _simple_df("Produk bagus!")
        mock_analyze.side_effect = AuthenticationError("bad key")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.rows[0].status    == RowStatus.FAILED
        assert result.rows[0].error_msg is not None
        assert len(result.rows[0].error_msg) > 0

    @patch("core.batch_analyzer.analyze_feedback")
    def test_all_rows_fail(self, mock_analyze):
        df = _simple_df("A", "B")
        mock_analyze.side_effect = AnalysisError("server error")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 0
        assert result.failed    == 2

    @patch("core.batch_analyzer.analyze_feedback")
    def test_error_does_not_abort_subsequent_rows(self, mock_analyze):
        """The batch must process every row regardless of earlier failures."""
        df = _simple_df(*[f"Ulasan {i}" for i in range(5)])
        # Rows 0, 2, 4 fail; rows 1, 3 succeed.
        mock_analyze.side_effect = [
            ResponseParseError("bad json"),    # row 0
            _make_analysis("Ulasan 1"),         # row 1
            TransientError("timeout"),          # row 2
            _make_analysis("Ulasan 3"),         # row 3
            AuthenticationError("bad key"),     # row 4
        ]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert mock_analyze.call_count == 5
        assert result.succeeded == 2
        assert result.failed    == 3


class TestRunBatchProgress:
    """on_progress callback is called correctly."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_progress_called_once_per_row(self, mock_analyze):
        df = _simple_df("A", "B", "C")
        mock_analyze.side_effect = [_make_analysis(t) for t in ["A", "B", "C"]]
        calls: list[tuple[int, int]] = []

        run_batch(
            df, "review",
            batch_size=10,
            batch_delay_seconds=0,
            on_progress=lambda done, total: calls.append((done, total)),
        )

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    @patch("core.batch_analyzer.analyze_feedback")
    def test_progress_called_for_skipped_rows(self, mock_analyze):
        """Skipped rows must still trigger the progress callback."""
        df = _simple_df("Bagus", "", "Kurang")
        mock_analyze.side_effect = [
            _make_analysis("Bagus"),
            _make_analysis("Kurang"),
        ]
        calls: list[tuple[int, int]] = []

        run_batch(
            df, "review",
            batch_size=10,
            batch_delay_seconds=0,
            on_progress=lambda done, total: calls.append((done, total)),
        )

        assert len(calls) == 3  # one per row, including the skipped one


class TestRunBatchInvalidColumn:
    """Calling run_batch with a non-existent column raises ValueError."""

    def test_missing_column_raises(self):
        df = pd.DataFrame({"ulasan": ["Bagus"]})
        with pytest.raises(ValueError, match="'review'"):
            run_batch(df, "review", batch_size=10, batch_delay_seconds=0)


# ── results_to_dataframe tests ────────────────────────────────────────────────

class TestResultsToDataframe:
    """Merging original DataFrame with BatchResult."""

    def _make_batch_result_success(self, texts: list[str]) -> BatchResult:
        rows = [
            RowResult(
                row_index=i,
                status=RowStatus.SUCCESS,
                analysis=_make_analysis(t),
            )
            for i, t in enumerate(texts)
        ]
        result = BatchResult()
        result.rows = rows
        return result

    def test_merged_has_original_columns(self):
        df = pd.DataFrame({"review": ["Bagus"], "rating": ["5"]})
        batch = self._make_batch_result_success(["Bagus"])
        merged = results_to_dataframe(df, batch)
        assert "review" in merged.columns
        assert "rating" in merged.columns

    def test_merged_has_ai_columns(self):
        df = _simple_df("Bagus")
        batch = self._make_batch_result_success(["Bagus"])
        merged = results_to_dataframe(df, batch)
        for col in ["ai_sentiment", "ai_category", "ai_key_issue",
                    "ai_evidence", "ai_recommendation", "ai_status", "ai_error"]:
            assert col in merged.columns, f"Missing column: {col}"

    def test_success_row_has_populated_ai_columns(self):
        df = _simple_df("Bagus sekali!")
        batch = self._make_batch_result_success(["Bagus sekali!"])
        merged = results_to_dataframe(df, batch)
        assert merged.loc[0, "ai_sentiment"] == "Positive"
        assert merged.loc[0, "ai_status"]    == RowStatus.SUCCESS.value
        assert merged.loc[0, "ai_error"]     == ""

    def test_skipped_row_has_empty_ai_columns(self):
        df = _simple_df("")
        result = BatchResult()
        result.rows = [RowResult(row_index=0, status=RowStatus.SKIPPED)]
        merged = results_to_dataframe(df, result)
        assert merged.loc[0, "ai_sentiment"]  == ""
        assert merged.loc[0, "ai_status"]     == RowStatus.SKIPPED.value

    def test_failed_row_has_error_column(self):
        df = _simple_df("Produk bagus!")
        result = BatchResult()
        result.rows = [
            RowResult(
                row_index=0,
                status=RowStatus.FAILED,
                error_msg="Layanan AI tidak tersedia.",
            )
        ]
        merged = results_to_dataframe(df, result)
        assert merged.loc[0, "ai_status"] == RowStatus.FAILED.value
        assert "Layanan" in merged.loc[0, "ai_error"]

    def test_row_count_matches_input(self):
        df = _simple_df("A", "B", "C")
        batch = self._make_batch_result_success(["A", "B", "C"])
        merged = results_to_dataframe(df, batch)
        assert len(merged) == 3

    def test_mixed_statuses_all_rows_present(self):
        """SUCCESS + SKIPPED + FAILED must each appear exactly once."""
        df = _simple_df("Bagus", "", "Error")
        result = BatchResult()
        result.rows = [
            RowResult(row_index=0, status=RowStatus.SUCCESS, analysis=_make_analysis("Bagus")),
            RowResult(row_index=1, status=RowStatus.SKIPPED),
            RowResult(row_index=2, status=RowStatus.FAILED, error_msg="Gagal."),
        ]
        merged = results_to_dataframe(df, result)
        statuses = list(merged["ai_status"])
        assert RowStatus.SUCCESS.value in statuses
        assert RowStatus.SKIPPED.value in statuses
        assert RowStatus.FAILED.value  in statuses


# ── Column-name robustness tests ──────────────────────────────────────────────

class TestColumnNameRobustness:
    """run_batch must work for any column name: simple, spaced, or accented."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_simple_column_name(self, mock_analyze):
        """Column named 'review' — baseline case."""
        df = pd.DataFrame({"review": ["Produk bagus!"]})
        mock_analyze.return_value = _make_analysis("Produk bagus!")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 1
        mock_analyze.assert_called_once_with("Produk bagus!")

    @patch("core.batch_analyzer.analyze_feedback")
    def test_column_name_with_space(self, mock_analyze):
        """Column named 'review text' (space in name) must be iterated correctly."""
        df = pd.DataFrame({"review text": ["Pengiriman lambat."]})
        mock_analyze.return_value = _make_analysis("Pengiriman lambat.")

        result = run_batch(df, "review text", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 1
        mock_analyze.assert_called_once_with("Pengiriman lambat.")

    @patch("core.batch_analyzer.analyze_feedback")
    def test_column_name_with_uppercase_and_space(self, mock_analyze):
        """Column named 'Ulasan Pelanggan' must be iterated correctly."""
        df = pd.DataFrame({"Ulasan Pelanggan": ["Barang rusak saat diterima."]})
        mock_analyze.return_value = _make_analysis("Barang rusak saat diterima.")

        result = run_batch(df, "Ulasan Pelanggan", batch_size=10, batch_delay_seconds=0)

        assert result.succeeded == 1
        mock_analyze.assert_called_once_with("Barang rusak saat diterima.")


# ── NaN / blank detection tests ───────────────────────────────────────────────

class TestNanAndBlankDetection:
    """Exact boundary conditions for the skip-row logic."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_actual_nan_is_skipped(self, mock_analyze):
        """pandas float NaN must be treated as missing and skipped."""
        df = pd.DataFrame({"review": [float("nan")]})

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        mock_analyze.assert_not_called()
        assert result.skipped == 1

    @patch("core.batch_analyzer.analyze_feedback")
    def test_none_is_skipped(self, mock_analyze):
        """Python None must be treated as missing and skipped."""
        df = pd.DataFrame({"review": [None]})

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        mock_analyze.assert_not_called()
        assert result.skipped == 1

    @patch("core.batch_analyzer.analyze_feedback")
    def test_empty_string_is_skipped(self, mock_analyze):
        """Empty string must be skipped."""
        df = pd.DataFrame({"review": [""]})

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        mock_analyze.assert_not_called()
        assert result.skipped == 1

    @patch("core.batch_analyzer.analyze_feedback")
    def test_whitespace_only_is_skipped(self, mock_analyze):
        """Whitespace-only string must be skipped."""
        df = pd.DataFrame({"review": ["   \t\n"]})

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        mock_analyze.assert_not_called()
        assert result.skipped == 1

    @patch("core.batch_analyzer.analyze_feedback")
    def test_literal_string_nan_is_analysed(self, mock_analyze):
        """The string literal 'nan' must NOT be treated as missing — it is real text."""
        df = pd.DataFrame({"review": ["nan"]})
        mock_analyze.return_value = _make_analysis("nan")

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        mock_analyze.assert_called_once_with("nan")
        assert result.succeeded == 1
        assert result.skipped   == 0


# ── Per-row exception isolation tests ────────────────────────────────────────

class TestPerRowExceptionIsolation:
    """An unexpected exception on one row must not stop the batch."""

    @patch("core.batch_analyzer.analyze_feedback")
    def test_runtime_error_on_one_row_continues_batch(self, mock_analyze):
        """RuntimeError on row 0 must not prevent rows 1 and 2 from being processed."""
        df = _simple_df("Baris 1", "Baris 2", "Baris 3")
        mock_analyze.side_effect = [
            RuntimeError("unexpected crash"),
            _make_analysis("Baris 2"),
            _make_analysis("Baris 3"),
        ]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert result.failed    == 1
        assert result.succeeded == 2
        assert result.rows[0].status == RowStatus.FAILED
        assert result.rows[1].status == RowStatus.SUCCESS
        assert result.rows[2].status == RowStatus.SUCCESS

    @patch("core.batch_analyzer.analyze_feedback")
    def test_row_index_matches_dataframe_position(self, mock_analyze):
        """RowResult.row_index must equal the 0-based position in the DataFrame."""
        df = _simple_df("A", "B", "C")
        mock_analyze.side_effect = [_make_analysis(t) for t in ["A", "B", "C"]]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        for expected_idx, row_result in enumerate(result.rows):
            assert row_result.row_index == expected_idx, (
                f"Expected row_index={expected_idx}, got {row_result.row_index}"
            )

    @patch("core.batch_analyzer.analyze_feedback")
    def test_total_rows_equals_input_rows(self, mock_analyze):
        """len(BatchResult.rows) must equal len(df) regardless of skip/fail mix."""
        df = _simple_df("OK", "", "FAIL", "OK again")
        mock_analyze.side_effect = [
            _make_analysis("OK"),
            RuntimeError("crash"),
            _make_analysis("OK again"),
        ]

        result = run_batch(df, "review", batch_size=10, batch_delay_seconds=0)

        assert len(result.rows) == len(df) == 4
        assert result.succeeded == 2
        assert result.skipped   == 1
        assert result.failed    == 1


# ── File-hash identity tests ──────────────────────────────────────────────────

class TestFileHash:
    """_file_hash must distinguish files by content, not by name or size."""

    def test_same_content_produces_same_hash(self):
        from ui.batch_input import _file_hash
        raw = b"review\nBagus sekali!\n"
        assert _file_hash(raw) == _file_hash(raw)

    def test_different_content_produces_different_hash(self):
        from ui.batch_input import _file_hash
        raw_a = b"review\nBagus sekali!\n"
        raw_b = b"review\nPengiriman lambat!\n"
        # Same length by coincidence is irrelevant — content differs.
        assert _file_hash(raw_a) != _file_hash(raw_b)

    def test_same_name_and_size_different_content_detected(self):
        """
        Two files that happen to share the same name and byte-length but have
        different content must produce different hashes.
        """
        from ui.batch_input import _file_hash
        # Craft two byte strings of equal length with different content.
        raw_a = b"review\nAAAAAAAAAAAA\n"
        raw_b = b"review\nBBBBBBBBBBBB\n"
        assert len(raw_a) == len(raw_b), "Test setup: lengths must be equal"
        assert _file_hash(raw_a) != _file_hash(raw_b)

    def test_hash_is_hex_string(self):
        from ui.batch_input import _file_hash
        result = _file_hash(b"some content")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)


# ── Column-change state invalidation tests ────────────────────────────────────

class TestColumnChangeInvalidation:
    """
    When the user switches the feedback column after a completed run,
    the stale BatchResult and merged DataFrame must be cleared from
    session_state before the new run starts.
    """

    def _make_state_with_result(self, column: str) -> dict:
        """Return a session_state dict as if a successful run on *column* just finished."""
        from ui.batch_input import (
            _KEY_BATCH_COLUMN,
            _KEY_BATCH_DF,
            _KEY_BATCH_FILE_ID,
            _KEY_BATCH_MERGED,
            _KEY_BATCH_RESULT,
        )
        import pandas as pd
        from core.batch_analyzer import BatchResult, RowResult, RowStatus

        result = BatchResult()
        result.rows = [RowResult(row_index=0, status=RowStatus.SUCCESS,
                                 analysis=_make_analysis("Bagus!"))]
        return {
            _KEY_BATCH_DF:      pd.DataFrame({"col_a": ["Bagus!"], "col_b": ["X"]}),
            _KEY_BATCH_RESULT:  result,
            _KEY_BATCH_MERGED:  pd.DataFrame({"col_a": ["Bagus!"], "ai_status": ["Berhasil"]}),
            _KEY_BATCH_FILE_ID: "abc123",
            _KEY_BATCH_COLUMN:  column,
        }

    def test_different_column_clears_result(self):
        """Switching to a different column must wipe _KEY_BATCH_RESULT."""
        import streamlit as st
        from contextlib import ExitStack
        from unittest.mock import patch

        from ui.batch_input import (
            _KEY_BATCH_COLUMN,
            _KEY_BATCH_MERGED,
            _KEY_BATCH_RESULT,
            _render_batch_run,
        )

        state = self._make_state_with_result("col_a")
        df = state["batch_original_df"]  # type: ignore[index]

        # Simulate: user picks "col_b" (was "col_a" for the previous run).
        # Button is NOT clicked, so run_batch is not called.
        with ExitStack() as stack:
            stack.enter_context(patch.object(st, "session_state", state))
            stack.enter_context(patch.object(st, "caption"))
            stack.enter_context(patch.object(st, "button", return_value=False))
            stack.enter_context(patch.object(st, "subheader"))
            _render_batch_run(df, "col_b")

        assert state[_KEY_BATCH_RESULT] is None, (
            "Stale result should be cleared when the column changes"
        )
        assert state[_KEY_BATCH_MERGED] is None, (
            "Stale merged DataFrame should be cleared when the column changes"
        )

    def test_same_column_preserves_result(self):
        """Re-rendering with the same column must NOT clear the cached result."""
        import streamlit as st
        from contextlib import ExitStack
        from unittest.mock import patch

        from ui.batch_input import (
            _KEY_BATCH_RESULT,
            _render_batch_run,
        )

        state = self._make_state_with_result("col_a")
        prior_result = state["batch_result"]  # type: ignore[index]
        df = state["batch_original_df"]  # type: ignore[index]

        with ExitStack() as stack:
            stack.enter_context(patch.object(st, "session_state", state))
            stack.enter_context(patch.object(st, "caption"))
            stack.enter_context(patch.object(st, "button", return_value=False))
            stack.enter_context(patch.object(st, "subheader"))
            _render_batch_run(df, "col_a")

        assert state[_KEY_BATCH_RESULT] is prior_result, (
            "Result should be preserved when the same column is selected"
        )


# ── Unexpected exception handling tests ──────────────────────────────────────

class TestBatchRunUnexpectedException:
    """
    If run_batch() raises an unexpected exception (not an AnalysisError),
    the UI must recover gracefully: progress indicators cleared, error message
    shown, None returned, session state left clean.
    """

    def _run_batch_with_exception(self, exc: Exception) -> dict:
        """
        Call _render_batch_run with a mocked run_batch that raises *exc*.
        Returns the resulting session_state dict.
        """
        import streamlit as st
        from contextlib import ExitStack
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from ui.batch_input import (
            _KEY_BATCH_COLUMN,
            _KEY_BATCH_MERGED,
            _KEY_BATCH_RESULT,
            _render_batch_run,
        )

        df = pd.DataFrame({"review": ["Bagus!", "Kurang bagus."]})
        state = {
            _KEY_BATCH_RESULT: None,
            _KEY_BATCH_MERGED: None,
            _KEY_BATCH_COLUMN: None,
        }

        progress_mock = MagicMock()
        status_mock   = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch.object(st, "session_state", state))
            stack.enter_context(patch.object(st, "caption"))
            stack.enter_context(patch.object(st, "button", return_value=True))
            stack.enter_context(patch.object(st, "progress", return_value=progress_mock))
            stack.enter_context(patch.object(st, "empty",    return_value=status_mock))
            stack.enter_context(patch.object(st, "spinner",  return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)))
            stack.enter_context(patch.object(st, "error"))
            stack.enter_context(patch("ui.batch_input.run_batch", side_effect=exc))

            result = _render_batch_run(df, "review")

        return {"result": result, "state": state,
                "progress_empty_called": progress_mock.empty.called,
                "status_empty_called":   status_mock.empty.called}

    def test_unexpected_exception_returns_none(self):
        """_render_batch_run must return None when run_batch raises unexpectedly."""
        out = self._run_batch_with_exception(RuntimeError("unexpected crash"))
        assert out["result"] is None

    def test_unexpected_exception_clears_progress(self):
        """progress_bar.empty() must be called to remove the stuck progress bar."""
        out = self._run_batch_with_exception(RuntimeError("unexpected crash"))
        assert out["progress_empty_called"], "progress_bar.empty() should be called on error"

    def test_unexpected_exception_clears_status_text(self):
        """status_text.empty() must be called to remove the stuck status caption."""
        out = self._run_batch_with_exception(RuntimeError("unexpected crash"))
        assert out["status_empty_called"], "status_text.empty() should be called on error"

    def test_unexpected_exception_leaves_result_none(self):
        """session_state result must remain None after an unexpected exception."""
        from ui.batch_input import _KEY_BATCH_RESULT
        out = self._run_batch_with_exception(RuntimeError("unexpected crash"))
        assert out["state"][_KEY_BATCH_RESULT] is None
