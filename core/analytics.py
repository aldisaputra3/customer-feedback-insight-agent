"""
core/analytics.py — Dashboard statistics for batch analysis results.

Responsibilities
────────────────
• Compute status counts (total / succeeded / skipped / failed).
• Compute sentiment and category distributions from SUCCESSFUL rows only.
• Filter the merged results for display without mutating the source data.

All functions are pure: they take the DataFrame produced by
core.batch_analyzer.results_to_dataframe() and never call the AI service.
This module must NOT import Streamlit or any network library.

PUBLIC API
──────────
    from core.analytics import compute_dashboard_stats, filter_results

    stats = compute_dashboard_stats(merged_df)
    view  = filter_results(merged_df, sentiments=["Negative"], categories=[])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from core.batch_analyzer import RowStatus
from llm.schemas import Category, Sentiment

SENTIMENT_ORDER: list[str] = [s.value for s in Sentiment]
CATEGORY_ORDER: list[str] = [c.value for c in Category]

_STATUS_COL = "ai_status"
_SENTIMENT_COL = "ai_sentiment"
_CATEGORY_COL = "ai_category"


@dataclass(frozen=True)
class DashboardStats:
    """
    Aggregated statistics for one batch run.

    sentiment_counts : DataFrame[label, count, percent] — one row per sentiment,
                       always in SENTIMENT_ORDER (zero counts included).
    category_counts  : DataFrame[label, count, percent] — one row per category,
                       always in CATEGORY_ORDER (zero counts included).
    Percentages are relative to *succeeded* and sum to ~100 when succeeded > 0.
    """
    total: int
    succeeded: int
    skipped: int
    failed: int
    sentiment_counts: pd.DataFrame
    category_counts: pd.DataFrame

    @property
    def has_success(self) -> bool:
        return self.succeeded > 0


def successful_rows(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows whose ai_status is SUCCESS."""
    if _STATUS_COL not in merged_df.columns:
        return merged_df.iloc[0:0]
    return merged_df[merged_df[_STATUS_COL] == RowStatus.SUCCESS.value]


def _distribution(values: pd.Series, order: list[str], denominator: int) -> pd.DataFrame:
    counts = values.value_counts()
    rows = []
    for label in order:
        count = int(counts.get(label, 0))
        percent = round(count / denominator * 100, 1) if denominator else 0.0
        rows.append({"label": label, "count": count, "percent": percent})
    return pd.DataFrame(rows, columns=["label", "count", "percent"])


def compute_dashboard_stats(merged_df: pd.DataFrame) -> DashboardStats:
    """
    Compute dashboard statistics from a results_to_dataframe() output.

    Skipped and failed rows count toward total/skipped/failed only — they are
    never included in the sentiment or category distributions.
    """
    status = (
        merged_df[_STATUS_COL]
        if _STATUS_COL in merged_df.columns
        else pd.Series([], dtype=str)
    )
    ok = successful_rows(merged_df)
    succeeded = len(ok)

    empty = pd.Series([], dtype=str)
    sentiments = ok[_SENTIMENT_COL] if _SENTIMENT_COL in ok.columns else empty
    categories = ok[_CATEGORY_COL] if _CATEGORY_COL in ok.columns else empty

    return DashboardStats(
        total=len(merged_df),
        succeeded=succeeded,
        skipped=int((status == RowStatus.SKIPPED.value).sum()),
        failed=int((status == RowStatus.FAILED.value).sum()),
        sentiment_counts=_distribution(sentiments, SENTIMENT_ORDER, succeeded),
        category_counts=_distribution(categories, CATEGORY_ORDER, succeeded),
    )


def filter_results(
    merged_df: pd.DataFrame,
    sentiments: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Return a filtered *copy* of merged_df for display.

    • No filter selected (both empty/None) → all rows, including skipped/failed.
    • Any filter selected → only successful rows matching every active filter
      (skipped/failed rows have no sentiment/category, so they cannot match).

    The input DataFrame is never modified.
    """
    sentiments = list(sentiments or [])
    categories = list(categories or [])

    if not sentiments and not categories:
        return merged_df.copy()

    view = successful_rows(merged_df)
    if sentiments:
        view = view[view[_SENTIMENT_COL].isin(sentiments)]
    if categories:
        view = view[view[_CATEGORY_COL].isin(categories)]
    return view.copy()
