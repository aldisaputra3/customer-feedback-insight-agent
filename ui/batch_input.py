"""
ui/batch_input.py — Batch CSV upload and analysis UI.

Responsibilities
────────────────
• Render the CSV file uploader and validate the uploaded file.
• Show a preview of the uploaded data.
• Let the user pick which column contains the feedback text.
• Start the batch analysis and display a live progress bar.
• Show a summary (succeeded / skipped / failed) after the run.
• Display the dashboard, filtered results table and export buttons
  (delegated to ui/analytics.py).
• Handle all error states with clear, non-technical messages.

This module must NOT import the OpenAI SDK directly.
All AI calls are delegated to core.batch_analyzer.run_batch.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from config.settings import get_settings
from core.batch_analyzer import (
    BatchResult,
    CsvValidationError,
    load_csv,
    results_to_dataframe,
    run_batch,
)
from ui.analytics import (
    clear_export_cache,
    render_dashboard,
    render_export,
    render_filtered_table,
)

# ── Session-state keys ────────────────────────────────────────────────────────

_KEY_BATCH_DF       = "batch_original_df"       # uploaded DataFrame
_KEY_BATCH_RESULT   = "batch_result"             # BatchResult after run
_KEY_BATCH_MERGED   = "batch_merged_df"          # merged results DataFrame
_KEY_BATCH_FILE_ID  = "batch_file_id"            # SHA-256 hash of file content
_KEY_BATCH_COLUMN   = "batch_last_run_column"    # column used for the last run


def _init_state() -> None:
    for key, default in [
        (_KEY_BATCH_DF,      None),
        (_KEY_BATCH_RESULT,  None),
        (_KEY_BATCH_MERGED,  None),
        (_KEY_BATCH_FILE_ID, None),
        (_KEY_BATCH_COLUMN,  None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ── Upload section ────────────────────────────────────────────────────────────

def _file_hash(raw: bytes) -> str:
    """Return the SHA-256 hex digest of *raw* for content-based identity."""
    return hashlib.sha256(raw).hexdigest()


def _render_upload() -> pd.DataFrame | None:
    """
    Render the file uploader and return the parsed DataFrame on success.
    Returns None when no valid file is available.

    File identity is based on the SHA-256 hash of the file contents so that
    two files with the same name and size but different content are correctly
    detected as distinct files.
    """
    settings = get_settings()

    st.subheader("📂 Unggah File CSV")
    st.caption(
        f"Format yang didukung: CSV (UTF-8 atau Latin-1). "
        f"Batas: {settings.max_csv_size_mb} MB, {settings.max_csv_rows} baris."
    )

    uploaded = st.file_uploader(
        label="Pilih file CSV",
        type=["csv"],
        label_visibility="collapsed",
        key="batch_csv_uploader",
    )

    if uploaded is None:
        return None

    # Read once — Streamlit UploadedFile can only be read once per re-run.
    raw = uploaded.read()

    # Content-based identity: SHA-256 of the file bytes.
    file_id = _file_hash(raw)

    if st.session_state[_KEY_BATCH_FILE_ID] != file_id:
        # New or changed file — reset all derived state.
        st.session_state[_KEY_BATCH_DF]      = None
        st.session_state[_KEY_BATCH_RESULT]  = None
        st.session_state[_KEY_BATCH_MERGED]  = None
        st.session_state[_KEY_BATCH_COLUMN]  = None
        st.session_state[_KEY_BATCH_FILE_ID] = file_id
        clear_export_cache()

    # Only parse once per file identity.
    if st.session_state[_KEY_BATCH_DF] is not None:
        return st.session_state[_KEY_BATCH_DF]

    try:
        df = load_csv(
            raw,
            max_size_bytes=settings.max_csv_size_bytes,
            max_rows=settings.max_csv_rows,
        )
        st.session_state[_KEY_BATCH_DF] = df
        return df
    except CsvValidationError as exc:
        st.error(f"⚠️ **File tidak valid:** {exc}", icon="⚠️")
        return None
    except Exception:
        st.error(
            "⚠️ **Gagal membaca file CSV.** "
            "Pastikan file tidak rusak dan coba lagi.",
            icon="⚠️",
        )
        return None


# ── Preview section ───────────────────────────────────────────────────────────

def _render_preview(df: pd.DataFrame) -> None:
    """Show the first few rows of the uploaded CSV."""
    st.markdown(
        f"**{len(df)} baris** · **{len(df.columns)} kolom**: "
        + ", ".join(f"`{c}`" for c in df.columns)
    )
    with st.expander("👁️ Pratinjau data (5 baris pertama)", expanded=True):
        st.dataframe(df.head(5), width="stretch", hide_index=True)


# ── Column selector ───────────────────────────────────────────────────────────

def _render_column_selector(df: pd.DataFrame) -> str | None:
    """
    Let the user choose which column contains the feedback text.

    Auto-selects a column whose name contains common keywords so the
    selector is pre-filled for typical CSVs without locking the user in.
    """
    columns = list(df.columns)

    _FEEDBACK_KEYWORDS = ("review", "feedback", "ulasan", "comment", "komentar", "text", "teks")
    default_idx = 0
    for i, col in enumerate(columns):
        if any(kw in col.lower() for kw in _FEEDBACK_KEYWORDS):
            default_idx = i
            break

    st.subheader("🏷️ Pilih Kolom Ulasan")
    selected = st.selectbox(
        label="Kolom yang berisi teks ulasan pelanggan:",
        options=columns,
        index=default_idx,
        key="batch_feedback_column",
    )
    return selected


# ── Batch runner with progress ────────────────────────────────────────────────

def _render_batch_run(df: pd.DataFrame, feedback_column: str) -> BatchResult | None:
    """
    Render the Analyze button, run the batch with a live progress bar,
    and return the BatchResult.  Returns None when the button has not been
    clicked or the run was not triggered this Streamlit re-run.

    If the user selected a different column from the previous run, the stale
    result is cleared so it is not displayed alongside the new column choice.
    """
    settings = get_settings()

    # Clear stale results when the user switches to a different column.
    if (
        st.session_state[_KEY_BATCH_RESULT] is not None
        and st.session_state[_KEY_BATCH_COLUMN] != feedback_column
    ):
        st.session_state[_KEY_BATCH_RESULT] = None
        st.session_state[_KEY_BATCH_MERGED] = None

    # Count how many rows will actually be sent to the API.
    non_blank = df[feedback_column].dropna().apply(lambda x: str(x).strip()).ne("").sum()
    st.caption(
        f"**{non_blank}** ulasan akan dianalisis · "
        f"**{len(df) - non_blank}** baris kosong akan dilewati."
    )

    if not st.button(
        "🚀 Mulai Analisis Batch",
        type="primary",
        width="stretch",
        key="batch_run_button",
    ):
        return None

    # Reset previous results before the new run.
    st.session_state[_KEY_BATCH_RESULT] = None
    st.session_state[_KEY_BATCH_MERGED] = None
    clear_export_cache()

    progress_bar = st.progress(0, text="Mempersiapkan analisis…")
    status_text  = st.empty()

    def on_progress(processed: int, total_rows: int) -> None:
        pct = processed / total_rows if total_rows else 1.0
        progress_bar.progress(
            min(pct, 1.0),
            text=f"Memproses baris {processed} dari {total_rows}…",
        )
        status_text.caption(f"⏳ {processed}/{total_rows} baris selesai diproses.")

    try:
        with st.spinner("🔄 Analisis sedang berjalan…"):
            batch_result = run_batch(
                df=df,
                feedback_column=feedback_column,
                batch_size=settings.batch_size,
                batch_delay_seconds=settings.batch_delay_seconds,
                on_progress=on_progress,
            )
    except Exception:
        # Unexpected error from run_batch — clean up UI state and inform the user.
        progress_bar.empty()
        status_text.empty()
        st.error(
            "❌ **Terjadi kesalahan tidak terduga saat menjalankan analisis batch.** "
            "Silakan coba kembali. Jika masalah berlanjut, periksa konfigurasi Anda.",
            icon="❌",
        )
        return None

    progress_bar.progress(1.0, text="✅ Analisis selesai!")
    status_text.empty()

    st.session_state[_KEY_BATCH_RESULT] = batch_result
    st.session_state[_KEY_BATCH_MERGED] = results_to_dataframe(df, batch_result)
    st.session_state[_KEY_BATCH_COLUMN] = feedback_column

    return batch_result


# ── Summary card ──────────────────────────────────────────────────────────────

def _render_summary(result: BatchResult) -> None:
    """Display status alerts for the run (counts are shown in the dashboard)."""
    st.divider()

    if result.succeeded == 0 and result.failed > 0:
        st.error(
            "**Semua analisis gagal.** "
            "Periksa API key, model, dan koneksi internet Anda, lalu coba kembali.",
            icon="❌",
        )
    elif result.failed > 0:
        st.warning(
            f"**{result.failed} ulasan gagal dianalisis.** "
            "Detail kegagalan tersedia di kolom `ai_error` pada tabel di bawah "
            "(kosongkan filter agar baris gagal ikut tampil).",
            icon="⚠️",
        )
    elif result.skipped > 0:
        st.info(
            f"**{result.skipped} baris dilewati** karena kolom ulasan kosong.",
            icon="ℹ️",
        )

    if result.succeeded > 0:
        st.success(
            f"**{result.succeeded} ulasan berhasil dianalisis.**",
            icon="✅",
        )


# ── Public entry point ────────────────────────────────────────────────────────

def render_batch_input_section() -> None:
    """
    Top-level function called by app.py.

    Orchestrates: upload → preview → column select → run → summary →
    dashboard → filtered table → export.
    """
    _init_state()

    df = _render_upload()
    if df is None:
        st.info(
            "📂 Unggah file CSV yang berisi ulasan pelanggan untuk memulai analisis batch.",
            icon="ℹ️",
        )
        return

    _render_preview(df)
    st.divider()

    feedback_column = _render_column_selector(df)
    if feedback_column is None:
        return

    st.divider()
    batch_result = _render_batch_run(df, feedback_column)

    # Display persisted state (result or error from this or the previous run).
    result_to_show = batch_result or st.session_state[_KEY_BATCH_RESULT]
    merged_to_show = st.session_state[_KEY_BATCH_MERGED]

    if result_to_show is not None:
        _render_summary(result_to_show)
    if merged_to_show is not None:
        # M5: dashboard, filters and export reuse the stored DataFrame —
        # no AI call happens here.
        render_dashboard(merged_to_show)
        st.divider()
        render_filtered_table(merged_to_show)
        st.divider()
        render_export(merged_to_show)
