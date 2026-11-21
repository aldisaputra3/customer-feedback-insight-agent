# llm package — AI/LLM integration layer
#
# Public re-exports for convenient top-level imports.
# Example:
#   from llm import FeedbackAnalysis, analyse_feedback

from llm.schemas import Category, FeedbackAnalysis, Sentiment
from llm.service import (
    AnalysisError,
    AuthenticationError,
    InvalidRequestError,
    ResponseParseError,
    TransientError,
    analyse_feedback,
)

__all__ = [
    # Schemas
    "FeedbackAnalysis",
    "Sentiment",
    "Category",
    # Service
    "analyse_feedback",
    # Exceptions
    "AnalysisError",
    "TransientError",
    "AuthenticationError",
    "InvalidRequestError",
    "ResponseParseError",
]
