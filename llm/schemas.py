"""
llm/schemas.py — Pydantic v2 response models for LLM-structured output.

This module defines the exact contract that every successful analysis call
must satisfy.  It is intentionally free of any OpenAI / HTTP imports so it
can be tested and imported without side effects.

OUTPUT CONTRACT
───────────────
Every call to the LLM must return exactly these six fields:

  original_text   – the customer review, verbatim.
  sentiment       – Positive | Negative | Neutral
  category        – one of seven predefined categories
  key_issue       – concise explanation of the main feedback (Bahasa Indonesia)
  evidence        – a direct quotation from original_text
  recommendation  – actionable advice grounded in the feedback (Bahasa Indonesia)

USAGE
─────
    from llm.schemas import FeedbackAnalysis, Sentiment, Category

    result = FeedbackAnalysis.model_validate(raw_dict)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Controlled vocabulary ─────────────────────────────────────────────────────

class Sentiment(str, Enum):
    """The three sentiment labels accepted by the output contract."""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL  = "Neutral"


class Category(str, Enum):
    """
    Seven predefined feedback categories.

    Use OTHER when the feedback is genuinely outside all named categories.
    """
    PRODUCT_QUALITY      = "Product Quality"
    PRICE                = "Price"
    DELIVERY             = "Delivery"
    CUSTOMER_SERVICE     = "Customer Service"
    PAYMENT              = "Payment"
    APP_USABILITY        = "Application / Usability"
    OTHER                = "Other"


# ── Response model ────────────────────────────────────────────────────────────

class FeedbackAnalysis(BaseModel):
    """
    Validated result of analysing a single customer-feedback text.

    All six fields are required; extra fields are forbidden.
    Pydantic will raise ValidationError if any field is missing, has the
    wrong type, or if an unexpected key is present.
    """

    model_config = {"extra": "forbid"}

    original_text: str = Field(
        description="The customer review text, preserved verbatim.",
    )
    sentiment: Sentiment = Field(
        description="Overall sentiment: Positive, Negative, or Neutral.",
    )
    category: Category = Field(
        description="Primary feedback category.",
    )
    key_issue: str = Field(
        description="Concise explanation of the main issue (Bahasa Indonesia).",
    )
    evidence: str = Field(
        description="Direct quotation from original_text that supports key_issue.",
    )
    recommendation: str = Field(
        description="Actionable recommendation grounded in the feedback (Bahasa Indonesia).",
    )

    # ── Field-level validators ────────────────────────────────────────────────

    @field_validator("original_text", "key_issue", "evidence", "recommendation")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return v

    # ── Cross-field validator ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def evidence_must_be_substring_of_original(self) -> "FeedbackAnalysis":
        """
        Guard against hallucinated evidence.

        Requires evidence to be an exact (case-sensitive) substring of
        original_text so that capitalisation and punctuation differences
        are caught.  The model is instructed to copy evidence verbatim, so
        an exact match is the correct contract.
        """
        if self.evidence not in self.original_text:
            raise ValueError(
                "evidence must be a substring of original_text. "
                "Do not invent quotations that are not in the original review."
            )
        return self


# ── JSON Schema for Groq Structured Outputs ───────────────────────────────────

def build_json_schema() -> JSONSchema:
    """
    Return a JSON Schema dict suitable for Groq's response_format parameter.

    The schema uses strict mode requirements:
      - All properties are listed in "required".
      - "additionalProperties" is False for every object.
      - Enum values are enumerated explicitly.

    This schema is generated from the Pydantic model and then patched to
    add the strict-mode constraints that Groq requires.
    """
    schema: dict[str, Any] = {
        "name": "feedback_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "original_text": {"type": "string"},
                "sentiment":     {"type": "string", "enum": [s.value for s in Sentiment]},
                "category":      {"type": "string", "enum": [c.value for c in Category]},
                "key_issue":     {"type": "string"},
                "evidence":      {"type": "string"},
                "recommendation":{"type": "string"},
            },
            "required": [
                "original_text",
                "sentiment",
                "category",
                "key_issue",
                "evidence",
                "recommendation",
            ],
            "additionalProperties": False,
        },
    }
    return schema
