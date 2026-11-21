"""
core/batch_analyzer.py — Batch CSV analysis business logic.

Responsibilities
────────────────
• Load a CSV from a file-like object and validate its structure.
• Let the caller choose which column contains the feedback text.
• Iterate over rows sequentially, calling analyze_feedback() for each.
• Continue past per-row failures so a bad row never aborts the whole batch.
• Return a BatchResult containing one RowResult per input row.

This module must NOT import the OpenAI SDK, Streamlit, or any HTTP library.
All AI calls are delegated to core.analyzer.analyze_feedback.

PUBLIC API
──────────
    from core.batch_analyzer import run_batch, load_csv, BatchResult, RowResult, RowStatus

    df = load_csv(uploaded_file)
    result = run_batch(df, feedback_column="review", on_progress=callback)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Callable

import pandas as pd

from core.analyzer import analyze_feedback
from llm.schemas import FeedbackAnalysis

logger = logging.getLogger(__name__)


# ── Status vocabulary ─────────────────────────────────────────────────────────

class RowStatus(str, Enum):
    """Processing outcome for a single CSV row."""
    SUCCESS = "Berhasil"
    SKIPPED = "Dilewati"   # blank / whitespace-only feedback text
    FAILED  = "Gagal"      # API or parse error


# ── Per-row result ────────────────────────────────────────────────────────────

@dataclass
class RowResult:
    """
    Analysis outcome for one CSV row.

    Attributes
    ----------
    row_index   : 0-based position of this row in the original DataFrame.
    status      : RowStatus.SUCCESS | SKIPPED | FAILED
    analysis    : Populated on SUCCESS; None otherwise.
    error_msg   : Human-readable failure reason on FAILED; None otherwise.
    """
    row_index: int
    status: RowStatus
    analysis: FeedbackAnalysis | None = None
    error_msg: str | None = None


# ── Batch result container ────────────────────────────────────────────────────

@dataclass
class BatchResult:
    """
    Aggregate result of a batch analysis run.

    Attributes
    ----------
    rows        : One RowResult per input row (in original order).
    total       : Total rows processed (excluding nothing — includes all statuses).
    succeeded   : Rows that produced a valid FeedbackAnalysis.
    skipped     : Rows with blank feedback text, not sent to the API.
    failed      : Rows where the API call or parsing failed.
    """
    rows: list[RowResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.rows if r.status == RowStatus.SUCCESS)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.rows if r.status == RowStatus.SKIPPED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r.status == RowStatus.FAILED)


# ── CSV loader ────────────────────────────────────────────────────────────────

class CsvValidationError(ValueError):
    """Raised when the uploaded CSV cannot be used for batch analysis."""


def load_csv(
    source: bytes | BytesIO,
    max_size_bytes: int = 5 * 1024 * 1024,
    max_rows: int = 1000,
) -> pd.DataFrame:
    """
    Read and validate a CSV file from bytes or a BytesIO stream.

    Parameters
    ----------
    source          : Raw CSV bytes or a file-like BytesIO object.
    max_size_bytes  : Hard limit on file size (default 5 MB).
    max_rows        : Hard limit on number of data rows (default 1 000).

    Returns
    -------
    pd.DataFrame with the CSV contents, all columns as strings where
    possible.

    Raises
    ------
    CsvValidationError
        • File is empty.
        • File exceeds size or row limits.
        • File cannot be decoded as UTF-8 or latin-1.
        • File has no columns after parsing.
    """
    raw: bytes = source if isinstance(source, bytes) else source.read()

    if not raw:
        raise CsvValidationError("File CSV kosong. Silakan unggah file yang berisi data.")

    if len(raw) > max_size_bytes:
        limit_mb = max_size_bytes / (1024 * 1024)
        raise CsvValidationError(
            f"Ukuran file terlalu besar. Batas maksimum adalah {limit_mb:.0f} MB."
        )

    # Try UTF-8 first, fall back to latin-1 for Windows-generated CSVs.
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(BytesIO(raw), encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise CsvValidationError(
                f"Tidak dapat membaca file CSV: {exc}"
            ) from exc
    else:
        raise CsvValidationError(
            "Encoding file tidak dikenali. Simpan ulang file CSV dalam format UTF-8."
        )

    if len(df.columns) == 0:
        raise CsvValidationError(
            "File CSV tidak memiliki kolom. Pastikan baris pertama berisi nama kolom."
        )

    if len(df) == 0:
        raise CsvValidationError(
            "File CSV tidak memiliki baris data. Silakan unggah file yang berisi ulasan."
        )

    if len(df) > max_rows:
        raise CsvValidationError(
            f"File CSV memiliki terlalu banyak baris ({len(df)}). "
            f"Batas maksimum adalah {max_rows} baris."
        )

    return df


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    df: pd.DataFrame,
    feedback_column: str,
    batch_size: int = 5,
    batch_delay_seconds: float = 1.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """
    Analyse every row in *df[feedback_column]* sequentially.

    Parameters
    ----------
    df                   : DataFrame produced by load_csv().
    feedback_column      : Name of the column that holds the review text.
    batch_size           : After this many API calls, pause for
                           *batch_delay_seconds* to avoid rate-limit errors.
    batch_delay_seconds  : Seconds to sleep between batches (0 = no pause).
    on_progress          : Optional callback(processed_so_far, total) called
                           after each row so the UI can update a progress bar.

    Returns
    -------
    BatchResult with one RowResult per row.

    Notes
    -----
    • Rows with blank or NaN feedback text are marked SKIPPED — the API is
      never called for them, so they don't count against rate limits.
    • The string literal "nan" is treated as valid user text, not as missing.
    • If the API call raises any exception (including unexpected ones), the row
      is marked FAILED and processing continues with the next row.
    • The *feedback_column* must exist in *df*; callers are responsible for
      validating this before calling run_batch().
    """
    if feedback_column not in df.columns:
        raise ValueError(
            f"Kolom '{feedback_column}' tidak ditemukan dalam DataFrame. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    result = BatchResult()
    total = len(df)
    api_calls_since_last_pause = 0

    # Iterate directly over the column values — works for any column name
    # regardless of spaces, special characters, or non-identifier characters.
    for idx, raw_text in enumerate(df[feedback_column].tolist()):

        # ── Skip NaN / None / blank rows ──────────────────────────────────────
        # pd.isna() catches actual NaN/None; the strip() check catches whitespace.
        # The string literal "nan" is NOT treated as missing — it is real text.
        if pd.isna(raw_text) or not str(raw_text).strip():
            result.rows.append(RowResult(row_index=idx, status=RowStatus.SKIPPED))
            if on_progress:
                on_progress(idx + 1, total)
            continue

        feedback_text = str(raw_text).strip()

        # ── Rate-limit pause between batches ──────────────────────────────────
        if api_calls_since_last_pause >= batch_size and batch_delay_seconds > 0:
            logger.debug(
                "Batch pause: %.1fs after %d calls", batch_delay_seconds, batch_size
            )
            time.sleep(batch_delay_seconds)
            api_calls_since_last_pause = 0

        # ── Call the analyzer ─────────────────────────────────────────────────
        try:
            analysis = analyze_feedback(feedback_text)
            result.rows.append(
                RowResult(row_index=idx, status=RowStatus.SUCCESS, analysis=analysis)
            )
            api_calls_since_last_pause += 1
            logger.info("Row %d/%d: SUCCESS (sentiment=%s)", idx + 1, total, analysis.sentiment.value)

        except Exception as exc:
            # Catch all exceptions (AnalysisError, ValueError, and unexpected
            # errors alike) so one bad row never aborts the whole batch.
            # BaseException is intentionally not caught (KeyboardInterrupt, etc.).
            error_msg = _user_friendly_error(exc)
            result.rows.append(
                RowResult(row_index=idx, status=RowStatus.FAILED, error_msg=error_msg)
            )
            api_calls_since_last_pause += 1
            logger.warning("Row %d/%d: FAILED — %s", idx + 1, total, type(exc).__name__)

        if on_progress:
            on_progress(idx + 1, total)

    logger.info(
        "Batch complete: %d succeeded, %d skipped, %d failed",
        result.succeeded, result.skipped, result.failed,
    )
    return result


# ── Results → DataFrame ───────────────────────────────────────────────────────

_ANALYSIS_COLUMNS = [
    "ai_original_text",
    "ai_sentiment",
    "ai_category",
    "ai_key_issue",
    "ai_evidence",
    "ai_recommendation",
    "ai_status",
    "ai_error",
]


def results_to_dataframe(
    original_df: pd.DataFrame,
    batch_result: BatchResult,
) -> pd.DataFrame:
    """
    Merge the original CSV columns with AI analysis results.

    Each row in the returned DataFrame corresponds to a row in *original_df*.
    AI columns are prefixed with ``ai_`` to avoid collisions with existing
    column names.  Skipped and failed rows have empty AI columns.

    Parameters
    ----------
    original_df  : The DataFrame originally passed to run_batch().
    batch_result : The BatchResult returned by run_batch().

    Returns
    -------
    pd.DataFrame with all original columns plus the eight AI columns.
    """
    ai_rows: list[dict] = []

    for row_result in batch_result.rows:
        if row_result.status == RowStatus.SUCCESS and row_result.analysis is not None:
            a = row_result.analysis
            ai_rows.append({
                "ai_original_text":    a.original_text,
                "ai_sentiment":        a.sentiment.value,
                "ai_category":         a.category.value,
                "ai_key_issue":        a.key_issue,
                "ai_evidence":         a.evidence,
                "ai_recommendation":   a.recommendation,
                "ai_status":           RowStatus.SUCCESS.value,
                "ai_error":            "",
            })
        elif row_result.status == RowStatus.SKIPPED:
            ai_rows.append({
                "ai_original_text":    "",
                "ai_sentiment":        "",
                "ai_category":         "",
                "ai_key_issue":        "",
                "ai_evidence":         "",
                "ai_recommendation":   "",
                "ai_status":           RowStatus.SKIPPED.value,
                "ai_error":            "",
            })
        else:  # FAILED
            ai_rows.append({
                "ai_original_text":    "",
                "ai_sentiment":        "",
                "ai_category":         "",
                "ai_key_issue":        "",
                "ai_evidence":         "",
                "ai_recommendation":   "",
                "ai_status":           RowStatus.FAILED.value,
                "ai_error":            row_result.error_msg or "",
            })

    ai_df = pd.DataFrame(ai_rows, columns=_ANALYSIS_COLUMNS)
    return pd.concat([original_df.reset_index(drop=True), ai_df], axis=1)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _user_friendly_error(exc: Exception) -> str:
    """Map internal exceptions to short, non-technical user-facing messages."""
    from llm.service import (
        AuthenticationError,
        InvalidRequestError,
        ResponseParseError,
        TransientError,
    )

    if isinstance(exc, AuthenticationError):
        return "Autentikasi gagal — periksa LLM_API_KEY."
    if isinstance(exc, TransientError):
        return "Layanan AI tidak tersedia sementara (timeout/rate limit)."
    if isinstance(exc, InvalidRequestError):
        return "Konfigurasi model tidak valid."
    if isinstance(exc, ResponseParseError):
        msg = str(exc)
        if "max_completion_tokens" in msg or "LLM_MAX_TOKENS" in msg:
            return "Respons terpotong — naikkan LLM_MAX_TOKENS."
        return "Format respons AI tidak valid."
    if isinstance(exc, ValueError):
        return "Teks ulasan tidak boleh kosong."
    return "Kesalahan tidak terduga."
