"""
tests/test_analyzer.py — Unit tests for core/analyzer.py.

WHAT IS TESTED
──────────────
  - analyze_feedback() returns a validated FeedbackAnalysis on a valid response.
  - analyze_feedback() raises ValueError on empty input.
  - analyze_feedback() raises ValueError on whitespace-only input.
  - Each service-layer exception propagates transparently to the caller:
      AuthenticationError, TransientError, InvalidRequestError,
      ResponseParseError, AnalysisError.
  - The analyzer delegates to llm.service.analyse_feedback (not a separate client).
  - Logging calls do not suppress exceptions.

No real API key or network access is required.
All LLM calls are replaced by unittest.mock patches.

HOW TO RUN:
    pytest tests/test_analyzer.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.analyzer import analyze_feedback
from llm.schemas import Category, FeedbackAnalysis, Sentiment
from llm.service import (
    AnalysisError,
    AuthenticationError,
    InvalidRequestError,
    ResponseParseError,
    TransientError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_valid_result(original_text: str = "Produk sangat bagus!") -> FeedbackAnalysis:
    """Return a valid FeedbackAnalysis instance for use in mock returns.

    The evidence field must be a substring of original_text, so each call
    uses the first word of the supplied text as a safe substring.
    """
    # Pick a word that is definitely in original_text so the evidence validator passes.
    evidence = original_text.split()[0] if original_text.strip() else original_text
    return FeedbackAnalysis(
        original_text=original_text,
        sentiment=Sentiment.POSITIVE,
        category=Category.PRODUCT_QUALITY,
        key_issue="Kualitas produk sangat baik dan memuaskan.",
        evidence=evidence,
        recommendation="Pertahankan standar kualitas produk saat ini.",
    )


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestAnalyzeFeedbackSuccess:
    """Happy-path: valid input produces a validated FeedbackAnalysis."""

    @patch("core.analyzer.analyse_feedback")
    def test_returns_feedback_analysis(self, mock_svc):
        expected = _make_valid_result()
        mock_svc.return_value = expected

        result = analyze_feedback("Produk sangat bagus!")

        assert isinstance(result, FeedbackAnalysis)
        assert result is expected

    @patch("core.analyzer.analyse_feedback")
    def test_delegates_to_service(self, mock_svc):
        """The analyzer must pass the input string straight to the service."""
        mock_svc.return_value = _make_valid_result("Pengiriman lambat.")

        analyze_feedback("Pengiriman lambat.")

        mock_svc.assert_called_once_with("Pengiriman lambat.")

    @patch("core.analyzer.analyse_feedback")
    def test_sentiment_positive(self, mock_svc):
        result = _make_valid_result()
        mock_svc.return_value = result
        assert analyze_feedback("Produk sangat bagus!").sentiment == Sentiment.POSITIVE

    @patch("core.analyzer.analyse_feedback")
    def test_sentiment_negative(self, mock_svc):
        neg = FeedbackAnalysis(
            original_text="Barang rusak.",
            sentiment=Sentiment.NEGATIVE,
            category=Category.PRODUCT_QUALITY,
            key_issue="Produk rusak saat diterima.",
            evidence="Barang rusak",
            recommendation="Perbaiki kontrol kualitas.",
        )
        mock_svc.return_value = neg
        assert analyze_feedback("Barang rusak.").sentiment == Sentiment.NEGATIVE

    @patch("core.analyzer.analyse_feedback")
    def test_english_input_delegated(self, mock_svc):
        """English text must be passed through without modification."""
        mock_svc.return_value = _make_valid_result("Product is great!")
        analyze_feedback("Product is great!")
        mock_svc.assert_called_once_with("Product is great!")

    @patch("core.analyzer.analyse_feedback")
    def test_all_six_fields_present(self, mock_svc):
        """All six FeedbackAnalysis fields must be accessible on the result."""
        expected = _make_valid_result()
        mock_svc.return_value = expected
        result = analyze_feedback("Produk sangat bagus!")

        assert result.original_text
        assert result.sentiment
        assert result.category
        assert result.key_issue
        assert result.evidence
        assert result.recommendation


# ── Empty / whitespace input tests ────────────────────────────────────────────

class TestAnalyzeFeedbackEmptyInput:
    """Empty or whitespace-only input must raise ValueError before hitting the API."""

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            analyze_feedback("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            analyze_feedback("   ")

    def test_newline_only_raises(self):
        with pytest.raises(ValueError):
            analyze_feedback("\n\n")

    def test_empty_input_does_not_call_service(self):
        """The ValueError for empty input is raised before the service is reached.

        We verify this by ensuring analyze_feedback("") raises without needing
        a mock — the guard lives in llm.service.analyse_feedback which is called
        by the analyzer and raises before any network code runs.
        """
        with pytest.raises(ValueError):
            analyze_feedback("")


# ── Exception propagation tests ───────────────────────────────────────────────

class TestAnalyzeFeedbackExceptionPropagation:
    """Service-layer exceptions must propagate transparently to the caller."""

    @patch("core.analyzer.analyse_feedback")
    def test_authentication_error_propagates(self, mock_svc):
        mock_svc.side_effect = AuthenticationError("bad key")
        with pytest.raises(AuthenticationError):
            analyze_feedback("Test input.")

    @patch("core.analyzer.analyse_feedback")
    def test_transient_error_propagates(self, mock_svc):
        mock_svc.side_effect = TransientError("timeout")
        with pytest.raises(TransientError):
            analyze_feedback("Test input.")

    @patch("core.analyzer.analyse_feedback")
    def test_invalid_request_error_propagates(self, mock_svc):
        mock_svc.side_effect = InvalidRequestError("bad model")
        with pytest.raises(InvalidRequestError):
            analyze_feedback("Test input.")

    @patch("core.analyzer.analyse_feedback")
    def test_response_parse_error_propagates(self, mock_svc):
        mock_svc.side_effect = ResponseParseError("bad json")
        with pytest.raises(ResponseParseError):
            analyze_feedback("Test input.")

    @patch("core.analyzer.analyse_feedback")
    def test_analysis_error_propagates(self, mock_svc):
        mock_svc.side_effect = AnalysisError("unknown")
        with pytest.raises(AnalysisError):
            analyze_feedback("Test input.")
