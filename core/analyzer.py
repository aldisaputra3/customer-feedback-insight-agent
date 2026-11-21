"""
core/analyzer.py — Business-facing analysis interface.

This module is the single entry point that the rest of the application
(Streamlit UI, batch processing, tests) should call.  It deliberately
does NOT import the OpenAI SDK or any HTTP library directly — all network
calls are handled inside the ``llm`` package.

By keeping this layer clean, we can:
  • Swap the underlying LLM provider without touching any UI code.
  • Unit-test business logic without network access.
  • Add caching, logging, or feature flags here without polluting llm/.

PUBLIC API
──────────
    from core.analyzer import analyze_feedback

    result = analyze_feedback("Pengiriman sangat lambat!")
    print(result.sentiment)       # Sentiment.NEGATIVE
    print(result.recommendation)  # "Tingkatkan koordinasi..."
"""

from __future__ import annotations

import logging

from llm.schemas import FeedbackAnalysis
from llm.service import analyse_feedback  # all SDK imports live there

logger = logging.getLogger(__name__)


def analyze_feedback(feedback_text: str) -> FeedbackAnalysis:
    """
    Analyse a single customer-feedback string and return structured insights.

    This is the primary business entry point for feedback analysis.  It
    delegates to :func:`llm.service.analyse_feedback` and can be extended
    here (e.g. logging, caching, feature flags) without changing service.py.

    Args:
        feedback_text: The raw customer review string (Indonesian or English).
                       Must be non-empty.

    Returns:
        A validated :class:`~llm.schemas.FeedbackAnalysis` instance containing
        all six required fields.

    Raises:
        ValueError:                      feedback_text is empty.
        llm.service.AuthenticationError: API key is missing or invalid.
        llm.service.InvalidRequestError: Bad model/parameter configuration.
        llm.service.TransientError:      Network/rate-limit failure after retries.
        llm.service.ResponseParseError:  Model returned malformed output.

    Example::

        from core.analyzer import analyze_feedback
        result = analyze_feedback("Barang bagus tapi pengiriman lama.")
        print(result.model_dump_json(indent=2))
    """
    logger.info("Analysing feedback (length=%d chars)", len(feedback_text))
    result = analyse_feedback(feedback_text)
    logger.info(
        "Analysis complete: sentiment=%s, category=%s",
        result.sentiment.value,
        result.category.value,
    )
    return result
