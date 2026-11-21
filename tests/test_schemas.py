"""
tests/test_schemas.py — Unit tests for llm/schemas.py.

These tests verify:
  - Valid structured responses are accepted.
  - Every valid Sentiment label is accepted.
  - Every valid Category is accepted.
  - Invalid sentiment labels are rejected.
  - Invalid category values are rejected.
  - Each required field, when missing, causes a ValidationError.
  - Unexpected extra fields cause a ValidationError (extra="forbid").
  - original_text is preserved verbatim.
  - Evidence must be a substring of original_text.
  - Both Indonesian and English inputs work.

NO API KEY OR NETWORK ACCESS is required — all tests operate on dict data.

HOW TO RUN:
    pytest tests/test_schemas.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm.schemas import Category, FeedbackAnalysis, Sentiment, build_json_schema


# ── Fixtures / helpers ────────────────────────────────────────────────────────

VALID_ENGLISH = {
    "original_text": "The product quality is excellent, delivery was fast.",
    "sentiment":     "Positive",
    "category":      "Product Quality",
    "key_issue":     "Kualitas produk sangat baik dan pengiriman cepat.",
    "evidence":      "The product quality is excellent",
    "recommendation": "Pertahankan kualitas produk dan kecepatan pengiriman.",
}

VALID_INDONESIAN = {
    "original_text": "Barang bagus tapi pengiriman lambat 3 hari.",
    "sentiment":     "Negative",
    "category":      "Delivery",
    "key_issue":     "Pengiriman terlambat tiga hari dari jadwal.",
    "evidence":      "pengiriman lambat 3 hari",
    "recommendation": "Tingkatkan koordinasi dengan kurir untuk mempercepat pengiriman.",
}


def make_valid(**overrides) -> dict:
    """Return a copy of VALID_ENGLISH with *overrides* applied."""
    data = dict(VALID_ENGLISH)
    data.update(overrides)
    return data


# ── Valid response tests ───────────────────────────────────────────────────────

class TestValidResponses:
    """Verify that well-formed dicts are accepted."""

    def test_valid_english_response(self):
        result = FeedbackAnalysis.model_validate(VALID_ENGLISH)
        assert result.sentiment == Sentiment.POSITIVE
        assert result.category  == Category.PRODUCT_QUALITY

    def test_valid_indonesian_response(self):
        result = FeedbackAnalysis.model_validate(VALID_INDONESIAN)
        assert result.sentiment == Sentiment.NEGATIVE
        assert result.category  == Category.DELIVERY

    def test_original_text_preserved(self):
        result = FeedbackAnalysis.model_validate(VALID_ENGLISH)
        assert result.original_text == VALID_ENGLISH["original_text"]

    def test_original_text_preserved_indonesian(self):
        result = FeedbackAnalysis.model_validate(VALID_INDONESIAN)
        assert result.original_text == VALID_INDONESIAN["original_text"]


# ── Sentiment label tests ──────────────────────────────────────────────────────

class TestSentimentLabels:
    """Verify each valid Sentiment label is accepted."""

    @pytest.mark.parametrize("label", ["Positive", "Negative", "Neutral"])
    def test_valid_sentiment(self, label: str):
        data = make_valid(sentiment=label)
        result = FeedbackAnalysis.model_validate(data)
        assert result.sentiment.value == label

    def test_invalid_sentiment_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(make_valid(sentiment="Happy"))

    def test_invalid_sentiment_case_rejected(self):
        """Enum matching is case-sensitive."""
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(make_valid(sentiment="positive"))


# ── Category label tests ───────────────────────────────────────────────────────

class TestCategoryLabels:
    """Verify each valid Category is accepted."""

    @pytest.mark.parametrize("cat", [
        "Product Quality",
        "Price",
        "Delivery",
        "Customer Service",
        "Payment",
        "Application / Usability",
        "Other",
    ])
    def test_valid_category(self, cat: str):
        data = make_valid(category=cat)
        result = FeedbackAnalysis.model_validate(data)
        assert result.category.value == cat

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(make_valid(category="Shipping"))

    def test_invalid_category_case_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(make_valid(category="delivery"))


# ── Required-field tests ───────────────────────────────────────────────────────

class TestRequiredFields:
    """Verify that every missing field raises ValidationError."""

    @pytest.mark.parametrize("field", [
        "original_text",
        "sentiment",
        "category",
        "key_issue",
        "evidence",
        "recommendation",
    ])
    def test_missing_field_rejected(self, field: str):
        data = dict(VALID_ENGLISH)
        del data[field]
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(data)


# ── Extra-field test ───────────────────────────────────────────────────────────

class TestExtraFields:
    """extra="forbid" must reject unexpected keys."""

    def test_extra_field_rejected(self):
        data = make_valid(confidence=0.99)
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(data)

    def test_multiple_extra_fields_rejected(self):
        data = make_valid(confidence=0.99, source="bot")
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(data)


# ── Evidence validator tests ───────────────────────────────────────────────────

class TestEvidenceValidator:
    """evidence must be an EXACT (case-sensitive) substring of original_text."""

    def test_evidence_is_exact_substring(self):
        result = FeedbackAnalysis.model_validate(VALID_ENGLISH)
        assert result.evidence in result.original_text

    def test_fabricated_evidence_rejected(self):
        data = make_valid(evidence="words that never appeared in the review")
        with pytest.raises(ValidationError, match="evidence must be a substring"):
            FeedbackAnalysis.model_validate(data)

    def test_evidence_wrong_case_rejected(self):
        """Exact match required — wrong capitalisation must be rejected."""
        # original_text has "The product quality is excellent" (title case "The")
        data = make_valid(evidence="THE PRODUCT QUALITY IS EXCELLENT")
        with pytest.raises(ValidationError, match="evidence must be a substring"):
            FeedbackAnalysis.model_validate(data)

    def test_evidence_exact_case_accepted(self):
        """Evidence that matches character-for-character must be accepted."""
        # Use a portion that exists verbatim in VALID_ENGLISH["original_text"]
        data = make_valid(evidence="delivery was fast")
        result = FeedbackAnalysis.model_validate(data)
        assert result.evidence == "delivery was fast"

    def test_evidence_partial_overlap_rejected(self):
        """Evidence that is similar but not an exact substring is rejected."""
        # original_text: "The product quality is excellent, delivery was fast."
        # This has a trailing comma which is not in original_text alone:
        data = make_valid(evidence="excellent delivery")  # words not adjacent
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(data)


# ── Blank-field tests ──────────────────────────────────────────────────────────

class TestBlankFields:
    """Blank / whitespace-only text fields must be rejected."""

    @pytest.mark.parametrize("field", ["original_text", "key_issue", "recommendation"])
    def test_blank_string_rejected(self, field: str):
        data = make_valid(**{field: "   "})
        with pytest.raises(ValidationError):
            FeedbackAnalysis.model_validate(data)


# ── JSON schema tests ──────────────────────────────────────────────────────────

class TestBuildJsonSchema:
    """Verify the Groq structured-output schema has the right shape."""

    def test_schema_has_name(self):
        schema = build_json_schema()
        assert schema["name"] == "feedback_analysis"

    def test_schema_strict_mode(self):
        schema = build_json_schema()
        assert schema["strict"] is True

    def test_schema_all_fields_present(self):
        schema = build_json_schema()
        props = schema["schema"]["properties"]
        assert set(props.keys()) == {
            "original_text", "sentiment", "category",
            "key_issue", "evidence", "recommendation",
        }

    def test_schema_additional_properties_false(self):
        schema = build_json_schema()
        assert schema["schema"]["additionalProperties"] is False

    def test_schema_all_fields_required(self):
        schema = build_json_schema()
        required = schema["schema"]["required"]
        assert set(required) == {
            "original_text", "sentiment", "category",
            "key_issue", "evidence", "recommendation",
        }

    def test_schema_sentiment_enum_values(self):
        schema = build_json_schema()
        assert schema["schema"]["properties"]["sentiment"]["enum"] == [
            "Positive", "Negative", "Neutral"
        ]

    def test_schema_category_enum_values(self):
        schema = build_json_schema()
        cats = schema["schema"]["properties"]["category"]["enum"]
        assert "Product Quality" in cats
        assert "Application / Usability" in cats
        assert "Other" in cats
        assert len(cats) == 7
