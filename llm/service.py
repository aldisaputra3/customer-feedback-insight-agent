"""
llm/service.py — LLM call, response parsing, validation, and retry logic.

Responsibilities
────────────────
• Call the LLM with a structured-output request.
• Parse the raw JSON response.
• Validate the parsed dict against FeedbackAnalysis (Pydantic).
• Retry transient errors (timeout, network, rate-limit, server errors) with
  exponential backoff, bounded by Settings.llm_max_retries.
• Never retry authentication errors or invalid-request errors.
• Never log or expose the API key.
• Raise descriptive exceptions so callers can display clear error messages.

PUBLIC API
──────────
    from llm.service import analyse_feedback, AnalysisError

    result: FeedbackAnalysis = analyse_feedback("Produk bagus!")

EXCEPTIONS
──────────
    AnalysisError          — base class for all service-layer errors
    ├── TransientError     — retried internally; re-raised after max retries
    ├── AuthenticationError— bad API key; not retried
    ├── InvalidRequestError— bad model/params; not retried
    └── ResponseParseError — model returned unparseable / invalid JSON
"""

from __future__ import annotations

import json
import logging
import time

import openai
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import ValidationError

from config.settings import Settings, get_settings
from llm.client import make_client
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.schemas import FeedbackAnalysis, build_json_schema

logger = logging.getLogger(__name__)


# ── Custom exception hierarchy ────────────────────────────────────────────────

class AnalysisError(Exception):
    """Base class for all LLM service errors."""


class TransientError(AnalysisError):
    """
    Raised for timeout, network failures, rate-limits (429), and 5xx errors.
    The service retries these internally; if retries are exhausted it re-raises
    this exception so the caller can display a helpful message.
    """


class AuthenticationError(AnalysisError):
    """
    Raised when the API key is missing, invalid, or revoked (401/403).
    Not retried — fixing the key requires human action.
    """


class InvalidRequestError(AnalysisError):
    """
    Raised for 400-class errors that indicate a programming mistake
    (e.g. unknown model name, bad parameter value).
    Not retried.
    """


class ResponseParseError(AnalysisError):
    """
    Raised when the model's response is empty, not valid JSON,
    or fails Pydantic validation.
    Not retried — the response has already been received.
    """


# ── Retry helpers ─────────────────────────────────────────────────────────────

# Errors that warrant a retry (transient by nature).
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

_BASE_BACKOFF_SECONDS: float = 1.0   # first sleep duration
_BACKOFF_MULTIPLIER:   float = 2.0   # doubles each attempt


def _is_retryable(exc: openai.OpenAIError) -> bool:
    """Return True if *exc* represents a transient failure worth retrying."""
    if isinstance(exc, openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


def _sleep(seconds: float) -> None:  # pragma: no cover — replaced by mock in tests
    """Thin wrapper around time.sleep so tests can patch it cheaply."""
    time.sleep(seconds)


# ── Core analysis function ────────────────────────────────────────────────────

def analyse_feedback(
    feedback_text: str,
    *,
    client: openai.OpenAI | None = None,
    settings: Settings | None = None,
) -> FeedbackAnalysis:
    """
    Analyse a single customer-feedback string and return a validated result.

    Args:
        feedback_text: The raw customer review to analyse.  Must be non-empty.
        client:        Optional pre-built OpenAI client (for testing / DI).
                       Defaults to make_client().
        settings:      Optional Settings object.  Defaults to get_settings().

    Returns:
        A validated :class:`FeedbackAnalysis` instance.

    Raises:
        ValueError:           If *feedback_text* is empty or whitespace-only.
        AuthenticationError:  API key is invalid or missing.
        InvalidRequestError:  Bad model, bad parameters, or refused by policy.
        TransientError:       Timeout/rate-limit after all retries exhausted.
        ResponseParseError:   Model returned empty, non-JSON, or invalid JSON.
    """
    if not feedback_text or not feedback_text.strip():
        raise ValueError("feedback_text must not be empty.")

    if settings is None:
        settings = get_settings()
    if client is None:
        client = make_client(settings)

    max_attempts: int = max(1, settings.llm_max_retries + 1)
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            raw_result = _call_api(feedback_text, client, settings)
            return _parse_and_validate(raw_result, feedback_text)

        except (AuthenticationError, InvalidRequestError, ResponseParseError):
            # Non-retryable — propagate immediately.
            raise

        except TransientError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                sleep_secs = _BASE_BACKOFF_SECONDS * (_BACKOFF_MULTIPLIER ** attempt)
                logger.warning(
                    "Transient error on attempt %d/%d, retrying in %.1fs: %s",
                    attempt + 1, max_attempts, sleep_secs, type(exc).__name__,
                )
                _sleep(sleep_secs)
            else:
                logger.error(
                    "All %d attempt(s) exhausted. Last error: %s",
                    max_attempts, exc,
                )

    # Only reached when all retries are exhausted.
    raise last_exc  # type: ignore[misc]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call_api(
    feedback_text: str,
    client: openai.OpenAI,
    settings: Settings,
) -> str:
    """
    Send the structured-output request and return the raw content string.

    Maps all openai SDK exceptions onto the service-layer hierarchy.
    """
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(feedback_text)},
            ],
            response_format=ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=build_json_schema(),
            ),
            max_completion_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            reasoning_effort="low",
        )
    except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError) as exc:
        raise TransientError(str(exc)) from exc
    except openai.AuthenticationError as exc:
        raise AuthenticationError(
            "API authentication failed. Check your LLM_API_KEY."
        ) from exc
    except openai.PermissionDeniedError as exc:
        raise AuthenticationError(
            "API permission denied. Check that your API key has access to this model."
        ) from exc
    except openai.BadRequestError as exc:
        raise InvalidRequestError(str(exc)) from exc
    except openai.APIStatusError as exc:
        if _is_retryable(exc):
            raise TransientError(str(exc)) from exc
        raise InvalidRequestError(str(exc)) from exc
    except openai.OpenAIError as exc:
        raise AnalysisError(str(exc)) from exc

    # Extract content from the first choice.
    choice = response.choices[0] if response.choices else None
    if choice is None:
        raise ResponseParseError("The API returned an empty choices list.")

    # Reject incomplete generation — a finish_reason of "length" means the
    # model ran out of tokens before closing the JSON document.  Accepting
    # partial JSON would cause a downstream parse error with a confusing
    # message; raising here gives a clear, actionable diagnosis instead.
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise ResponseParseError(
            f"The model stopped mid-generation because max_completion_tokens was "
            f"reached before the JSON response was complete. "
            f"Current limit: {settings.llm_max_tokens} tokens. "
            f"Increase LLM_MAX_TOKENS in your .env file."
        )

    # Handle model refusals (e.g. content-policy rejection).
    if hasattr(choice.message, "refusal") and choice.message.refusal:
        raise ResponseParseError(
            f"The model refused to process this input: {choice.message.refusal}"
        )

    content = choice.message.content
    if not content or not content.strip():
        raise ResponseParseError("The model returned an empty response body.")

    return content


def _parse_and_validate(raw_content: str, original_text: str) -> FeedbackAnalysis:
    """
    Parse the raw JSON string from the model and validate it with Pydantic.

    Args:
        raw_content:   The string returned by the model.
        original_text: The original feedback text that was sent to the model.
                       Used both for the original_text integrity check and by
                       the evidence-substring validator inside FeedbackAnalysis.

    Returns:
        A validated :class:`FeedbackAnalysis`.

    Raises:
        ResponseParseError: JSON is malformed, original_text was altered,
                            or Pydantic validation fails.
    """
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(
            f"Model response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ResponseParseError(
            f"Expected a JSON object, got {type(data).__name__}."
        )

    # ── Issue 1: original_text integrity check ────────────────────────────────
    # The model must echo back the review verbatim.  Any modification
    # (translation, paraphrase, truncation, whitespace change) is rejected.
    returned_text = data.get("original_text")
    if returned_text != original_text:
        raise ResponseParseError(
            "Model altered original_text. The review must be echoed back without any modification."
        )

    try:
        return FeedbackAnalysis.model_validate(data)
    except ValidationError as exc:
        raise ResponseParseError(
            f"Model response failed schema validation: {exc}"
        ) from exc
