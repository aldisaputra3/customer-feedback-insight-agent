"""
ui/analytics.py — Dashboard, result filters, and export buttons for batch results.

Everything here reads the merged results DataFrame already stored in
st.session_state by ui/batch_input.py.  No function in this module calls the
AI service: rendering charts, changing filters, or downloading files only
re-uses data that is already in memory.

PUBLIC API
──────────
    render_dashboard(merged_df)        # KPI row + sentiment & category charts
    render_filtered_table(merged_df)   # sentiment/category filters + table
    render_export(merged_df)           # CSV / Excel download buttons
"""

from __future__ import annotations

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from core.analytics import (
    CATEGORY_ORDER,
    SENTIMENT_ORDER,
    DashboardStats,
    compute_dashboard_stats,
    filter_results,
)
from core.batch_analyzer import RowStatus
from core.exporter import CSV_MIME, EXCEL_MIME, to_csv_bytes, to_excel_bytes

_KEY_EXPORT_CACHE = "batch_export_cache"
_KEY_FILTER_SENTIMENT = "batch_filter_sentiment"
_KEY_FILTER_CATEGORY = "batch_filter_category"

_SENTIMENT_COLORS = {
    "Positive": "#2E9E5B",
    "Negative": "#D64545",
    "Neutral": "#8A8F98",
}
_CATEGORY_COLOR = "#3B6FD8"

_STATUS_ICON = {
    RowStatus.SUCCESS.value: "🟢",
    RowStatus.SKIPPED.value: "🟡",
    RowStatus.FAILED.value: "🔴",
}


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _sentiment_chart(stats: DashboardStats) -> alt.Chart:
    data = stats.sentiment_counts.assign(
        text=lambda d: d["count"].astype(str) + " (" + d["percent"].map("{:.1f}%".format) + ")"
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=SENTIMENT_ORDER, title=None),
        x=alt.X("count:Q", title="Jumlah ulasan", axis=alt.Axis(tickMinStep=1)),
        tooltip=[
            alt.Tooltip("label:N", title="Sentimen"),
            alt.Tooltip("count:Q", title="Jumlah"),
            alt.Tooltip("percent:Q", title="Persentase (%)", format=".1f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=list(_SENTIMENT_COLORS), range=list(_SENTIMENT_COLORS.values())
            ),
            legend=None,
        )
    )
    labels = base.mark_text(align="left", dx=4).encode(text="text:N")
    return (bars + labels).properties(height=150)


def _category_chart(stats: DashboardStats) -> alt.Chart:
    data = stats.category_counts.sort_values(
        ["count", "label"], ascending=[False, True]
    )
    order = data["label"].tolist()
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("count:Q", title="Jumlah feedback", axis=alt.Axis(tickMinStep=1)),
        tooltip=[
            alt.Tooltip("label:N", title="Kategori"),
            alt.Tooltip("count:Q", title="Jumlah"),
            alt.Tooltip("percent:Q", title="Persentase (%)", format=".1f"),
        ],
    )
    bars = base.mark_bar(color=_CATEGORY_COLOR, cornerRadiusEnd=3)
    labels = base.mark_text(align="left", dx=4).encode(text="count:Q")
    return (bars + labels).properties(height=260)


def render_dashboard(merged_df: pd.DataFrame) -> DashboardStats:
    """Render KPI metrics and distribution charts. Returns the computed stats."""
    stats = compute_dashboard_stats(merged_df)

    st.subheader("📈 Dashboard Analitik")

    with st.container(horizontal=True):
        st.metric("Total ulasan", stats.total, border=True)
        st.metric("✅ Berhasil", stats.succeeded, border=True)
        st.metric("⏭️ Dilewati", stats.skipped, border=True)
        st.metric("❌ Gagal", stats.failed, border=True)

    if not stats.has_success:
        st.info(
            "Belum ada ulasan yang berhasil dianalisis, sehingga distribusi "
            "sentimen dan kategori belum dapat ditampilkan.",
            icon="ℹ️",
        )
        return stats

    st.caption(
        f"Distribusi di bawah dihitung dari **{stats.succeeded}** ulasan yang berhasil "
        "dianalisis. Baris yang dilewati atau gagal tidak dihitung."
    )

    with st.container(border=True):
        st.markdown("**Distribusi sentimen**")
        st.altair_chart(_sentiment_chart(stats))
        with st.container(horizontal=True):
            for _, row in stats.sentiment_counts.iterrows():
                st.metric(row["label"], f"{row['percent']:.1f}%", help=f"{row['count']} ulasan")

    with st.container(border=True):
        st.markdown("**Distribusi kategori**")
        st.altair_chart(_category_chart(stats))

    return stats


# ── Filtered results table ────────────────────────────────────────────────────

def render_filtered_table(merged_df: pd.DataFrame) -> None:
    """Render sentiment/category filters and the (filtered) results table."""
    st.subheader("📋 Tabel Hasil Analisis")

    sentiments = st.pills(
        "Filter sentimen",
        options=SENTIMENT_ORDER,
        selection_mode="multi",
        key=_KEY_FILTER_SENTIMENT,
    )
    categories = st.multiselect(
        "Filter kategori",
        options=CATEGORY_ORDER,
        key=_KEY_FILTER_CATEGORY,
        placeholder="Semua kategori",
    )

    view = filter_results(merged_df, sentiments, categories)
    if sentiments or categories:
        st.caption(
            f"Menampilkan **{len(view)}** dari **{len(merged_df)}** baris "
            "(hanya ulasan berhasil yang cocok dengan filter). "
            "Filter tidak memengaruhi statistik dashboard maupun file ekspor."
        )

    if view.empty:
        st.info("Tidak ada ulasan yang cocok dengan filter yang dipilih.", icon="🔍")
        return

    display_df = view.copy()
    if "ai_status" in display_df.columns:
        display_df["ai_status"] = display_df["ai_status"].map(
            lambda s: f"{_STATUS_ICON.get(s, '')} {s}"
        )
    st.dataframe(display_df, hide_index=True)


# ── Export ────────────────────────────────────────────────────────────────────

def clear_export_cache() -> None:
    """Drop cached export bytes (called when a new file or run replaces results)."""
    st.session_state.pop(_KEY_EXPORT_CACHE, None)


def _export_bytes(merged_df: pd.DataFrame) -> tuple[bytes, bytes | None]:
    """
    Return (csv_bytes, xlsx_bytes), cached in session_state per DataFrame object
    so they are generated once per batch run, not on every rerun.

    xlsx_bytes is None when the Excel engine (openpyxl) is not installed, so a
    missing optional library never breaks the page or the CSV download.
    """
    cache = st.session_state.get(_KEY_EXPORT_CACHE)
    if cache is not None and cache["df"] is merged_df:
        return cache["csv"], cache["xlsx"]

    csv_bytes = to_csv_bytes(merged_df)
    try:
        xlsx_bytes: bytes | None = to_excel_bytes(merged_df)
    except ImportError:
        xlsx_bytes = None
    st.session_state[_KEY_EXPORT_CACHE] = {"df": merged_df, "csv": csv_bytes, "xlsx": xlsx_bytes}
    return csv_bytes, xlsx_bytes


def render_export(merged_df: pd.DataFrame) -> None:
    """Render CSV and Excel download buttons for the full (unfiltered) results."""
    st.subheader("💾 Unduh Hasil Analisis")
    st.caption(
        "File berisi seluruh baris (berhasil, dilewati, dan gagal) beserta semua "
        "kolom asli dan kolom hasil AI."
    )

    csv_bytes, xlsx_bytes = _export_bytes(merged_df)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    with st.container(horizontal=True):
        st.download_button(
            "Unduh CSV",
            data=csv_bytes,
            file_name=f"hasil_analisis_{stamp}.csv",
            mime=CSV_MIME,
            icon=":material/download:",
            on_click="ignore",
            key="batch_download_csv",
            width="stretch",
        )
        if xlsx_bytes is None:
            st.warning(
                "Ekspor Excel tidak tersedia karena library `openpyxl` belum "
                "terpasang. Jalankan `pip install -r requirements.txt` lalu muat ulang.",
                icon="⚠️",
            )
            return
        st.download_button(
            "Unduh Excel (.xlsx)",
            data=xlsx_bytes,
            file_name=f"hasil_analisis_{stamp}.xlsx",
            mime=EXCEL_MIME,
            icon=":material/table_view:",
            on_click="ignore",
            key="batch_download_xlsx",
            width="stretch",
        )
