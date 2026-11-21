"""
tests/test_manual_input.py — Unit tests for ui/manual_input._run_analysis caching.

WHAT IS TESTED
──────────────
  Caching — successful result:
    - Same text + existing success result → API NOT called again (cache hit).
    - Cached result is preserved in session_state after a cache hit.

  Retry after failure:
    - Same text + previous error → API IS called again (cache miss, allows retry).
    - After successful retry, result replaces the previous error in state.
    - After a failed retry, the new error message is stored (old one overwritten).

  New text:
    - Different text from a previous success → API IS called (cache miss).
    - Stale result is cleared from state before the new call begins.

  Error state transitions:
    - Each service-layer exception maps to a user-facing string in session_state.

APPROACH
────────
`_run_analysis` only touches `st.session_state` (a dict-like object) and calls
`core.analyzer.analyze_feedback`.  We inject a plain dict as `session_state` by
patching the `streamlit` module's `session_state` attribute and by patching
`st.spinner` with a no-op context manager.  `analyze_feedback` is patched at its
import site inside `ui.manual_input`.

HOW TO RUN:
    pytest tests/test_manual_input.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import streamlit as st

from llm.schemas import Category, FeedbackAnalysis, Sentiment
from llm.service import (
    AnalysisError,
    AuthenticationError,
    InvalidRequestError,
    ResponseParseError,
    TransientError,
)
from ui.manual_input import (
    _KEY_ERROR,
    _KEY_RESULT,
    _KEY_SUBMITTED,
    _run_analysis,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(
    submitted: str = "",
    result: FeedbackAnalysis | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a minimal session_state dict."""
    return {
        _KEY_SUBMITTED: submitted,
        _KEY_RESULT:    result,
        _KEY_ERROR:     error,
    }


def _make_valid_result(original_text: str = "Produk sangat bagus!") -> FeedbackAnalysis:
    """Return a valid FeedbackAnalysis that satisfies the evidence-substring rule."""
    evidence = original_text.split()[0]
    return FeedbackAnalysis(
        original_text=original_text,
        sentiment=Sentiment.POSITIVE,
        category=Category.PRODUCT_QUALITY,
        key_issue="Kualitas produk sangat baik.",
        evidence=evidence,
        recommendation="Pertahankan standar kualitas produk.",
    )


@contextmanager
def _noop_spinner(_msg: str = ""):
    """Replacement for st.spinner that does nothing."""
    yield


def _patch_all(state: dict[str, Any], analyze_side: Any = None, analyze_return: Any = None):
    """
    Return a context manager stack that patches:
      - st.session_state  → state dict
      - st.spinner        → _noop_spinner
      - ui.manual_input.analyze_feedback → MagicMock

    Usage::

        with _patch_all(state, analyze_return=result) as mock_analyze:
            _run_analysis("text")
        mock_analyze.assert_called_once_with("text")
    """
    from contextlib import ExitStack

    @contextmanager
    def _inner():
        with ExitStack() as stack:
            stack.enter_context(patch.object(st, "session_state", state))
            stack.enter_context(patch.object(st, "spinner", _noop_spinner))
            mock_analyze = stack.enter_context(
                patch("ui.manual_input.analyze_feedback")
            )
            if analyze_side is not None:
                mock_analyze.side_effect = analyze_side
            elif analyze_return is not None:
                mock_analyze.return_value = analyze_return
            yield mock_analyze

    return _inner()


# ── Cache-hit tests ───────────────────────────────────────────────────────────

class TestCacheHit:
    """Same text + prior success → API must NOT be called again."""

    def test_same_text_success_does_not_call_api(self):
        prior_result = _make_valid_result("Produk sangat bagus!")
        state = _make_state(submitted="Produk sangat bagus!", result=prior_result)

        with _patch_all(state) as mock_analyze:
            _run_analysis("Produk sangat bagus!")

        mock_analyze.assert_not_called()

    def test_same_text_success_preserves_cached_result(self):
        prior_result = _make_valid_result("Produk sangat bagus!")
        state = _make_state(submitted="Produk sangat bagus!", result=prior_result)

        with _patch_all(state):
            _run_analysis("Produk sangat bagus!")

        assert state[_KEY_RESULT] is prior_result


# ── Retry-after-failure tests ─────────────────────────────────────────────────

class TestRetryAfterFailure:
    """Same text + prior error → API MUST be called (retry allowed)."""

    def test_same_text_prior_error_calls_api(self):
        new_result = _make_valid_result("Produk sangat bagus!")
        state = _make_state(
            submitted="Produk sangat bagus!",
            error="🌐 **Layanan AI sementara tidak tersedia.**",
        )

        with _patch_all(state, analyze_return=new_result) as mock_analyze:
            _run_analysis("Produk sangat bagus!")

        mock_analyze.assert_called_once_with("Produk sangat bagus!")

    def test_retry_success_stores_result_and_clears_error(self):
        new_result = _make_valid_result("Produk sangat bagus!")
        state = _make_state(submitted="Produk sangat bagus!", error="previous error")

        with _patch_all(state, analyze_return=new_result):
            _run_analysis("Produk sangat bagus!")

        assert state[_KEY_RESULT] is new_result
        assert state[_KEY_ERROR] is None

    def test_retry_still_failing_stores_new_error(self):
        state = _make_state(submitted="Produk sangat bagus!", error="old error")

        with _patch_all(state, analyze_side=TransientError("still failing")):
            _run_analysis("Produk sangat bagus!")

        assert state[_KEY_RESULT] is None
        assert state[_KEY_ERROR] is not None
        # Must be a fresh user-facing string, not the raw exception message.
        err = state[_KEY_ERROR].lower()
        assert "rate limit" in err or "tidak tersedia" in err or "layanan" in err

    def test_error_state_does_not_block_third_attempt(self):
        """Repeated failures must never permanently lock the retry path."""
        state = _make_state(submitted="Teks uji.", error="error from second attempt")
        new_result = _make_valid_result("Teks uji.")

        with _patch_all(state, analyze_return=new_result) as mock_analyze:
            _run_analysis("Teks uji.")

        mock_analyze.assert_called_once()
        assert state[_KEY_RESULT] is new_result


# ── New-text tests ────────────────────────────────────────────────────────────

class TestNewText:
    """Different text → always call the API, stale state must be cleared first."""

    def test_different_text_calls_api(self):
        prior_result = _make_valid_result("Teks lama.")
        new_result = _make_valid_result("Teks baru.")
        state = _make_state(submitted="Teks lama.", result=prior_result)

        with _patch_all(state, analyze_return=new_result) as mock_analyze:
            _run_analysis("Teks baru.")

        mock_analyze.assert_called_once_with("Teks baru.")

    def test_different_text_clears_stale_result_before_call(self):
        """Stale result from a previous submission must be wiped before the
        new API call, so the user never sees old data alongside a spinner."""
        stale_result = _make_valid_result("Teks lama.")
        state = _make_state(submitted="Teks lama.", result=stale_result)

        call_order: list[str] = []

        def fake_analyze(text: str) -> FeedbackAnalysis:
            # By the time the API is called, the stale result must already be None.
            call_order.append("api_called")
            assert state[_KEY_RESULT] is None, (
                "Stale result was not cleared before the API call"
            )
            return _make_valid_result(text)

        with _patch_all(state, analyze_side=fake_analyze):
            _run_analysis("Teks baru.")

        assert "api_called" in call_order
        assert state[_KEY_RESULT] is not None

    def test_different_text_stores_new_result(self):
        prior_result = _make_valid_result("Teks lama.")
        new_result = _make_valid_result("Teks baru.")
        state = _make_state(submitted="Teks lama.", result=prior_result)

        with _patch_all(state, analyze_return=new_result):
            _run_analysis("Teks baru.")

        assert state[_KEY_RESULT] is new_result
        assert state[_KEY_SUBMITTED] == "Teks baru."


# ── Error-mapping tests ───────────────────────────────────────────────────────

class TestErrorMapping:
    """Each service exception must map to a user-facing string in session_state."""

    def _run_with_exception(self, exc: Exception) -> dict[str, Any]:
        state = _make_state()
        with _patch_all(state, analyze_side=exc):
            _run_analysis("Ulasan pelanggan.")
        return state

    def test_authentication_error_sets_error_state(self):
        state = self._run_with_exception(AuthenticationError("bad key"))
        assert state[_KEY_RESULT] is None
        assert state[_KEY_ERROR] is not None
        assert "API key" in state[_KEY_ERROR] or "Autentikasi" in state[_KEY_ERROR]

    def test_transient_error_sets_error_state(self):
        state = self._run_with_exception(TransientError("timeout"))
        assert state[_KEY_RESULT] is None
        assert state[_KEY_ERROR] is not None

    def test_invalid_request_error_sets_error_state(self):
        state = self._run_with_exception(InvalidRequestError("bad model"))
        assert state[_KEY_RESULT] is None
        assert state[_KEY_ERROR] is not None

    def test_response_parse_error_sets_error_state(self):
        state = self._run_with_exception(ResponseParseError("bad json"))
        assert state[_KEY_RESULT] is None
        assert state[_KEY_ERROR] is not None

    def test_token_limit_error_mentions_max_tokens(self):
        exc = ResponseParseError(
            "The model stopped mid-generation because max_completion_tokens was reached."
        )
        state = self._run_with_exception(exc)
        assert "LLM_MAX_TOKENS" in state[_KEY_ERROR] or "token" in state[_KEY_ERROR].lower()

    def test_analysis_error_sets_error_state(self):
        state = self._run_with_exception(AnalysisError("unknown"))
        assert state[_KEY_RESULT] is None
        assert state[_KEY_ERROR] is not None

    def test_error_state_does_not_contain_raw_exception_message(self):
        """Raw exception internals must not leak into the user-facing message."""
        exc = AuthenticationError("secret-internal-token-abc123")
        state = self._run_with_exception(exc)
        assert "secret-internal-token-abc123" not in state[_KEY_ERROR]


# ── Invalid-submission state-clear tests ─────────────────────────────────────

class TestInvalidSubmissionClearsState:
    """
    When the button is clicked but input fails validation, stale session state
    (result or error from a previous run) must be cleared.

    This prevents old analysis results from lingering on-screen alongside a
    fresh 'input is empty' or 'input too long' error message.
    """

    def _run_orchestrator_with_form_return(
        self,
        form_return: tuple[bool, str | None],
        initial_state: dict,
    ) -> dict:
        """
        Call render_manual_input_section() with a patched _render_input_form
        that returns `form_return`, and return the resulting session_state dict.

        All Streamlit rendering helpers are stubbed out so no runtime is needed.
        """
        from contextlib import ExitStack
        from unittest.mock import patch

        import streamlit as _st

        with ExitStack() as stack:
            stack.enter_context(patch.object(_st, "session_state", initial_state))
            stack.enter_context(
                patch("ui.manual_input._render_input_form", return_value=form_return)
            )
            stack.enter_context(patch("ui.manual_input._run_analysis"))
            stack.enter_context(patch("ui.manual_input._render_result"))
            stack.enter_context(patch("ui.manual_input._render_error"))
            stack.enter_context(patch("ui.manual_input._init_state"))
            from ui.manual_input import render_manual_input_section
            render_manual_input_section()

        return initial_state

    def test_empty_submission_clears_stale_result(self):
        """Stale result must be wiped when user submits empty input."""
        from llm.schemas import Category, FeedbackAnalysis, Sentiment

        prior_result = FeedbackAnalysis(
            original_text="Produk sangat bagus!",
            sentiment=Sentiment.POSITIVE,
            category=Category.PRODUCT_QUALITY,
            key_issue="Kualitas produk sangat baik.",
            evidence="Produk",
            recommendation="Pertahankan standar kualitas.",
        )
        state = _make_state(submitted="Produk sangat bagus!", result=prior_result)

        result_state = self._run_orchestrator_with_form_return((True, None), state)

        assert result_state[_KEY_RESULT] is None, (
            "Stale result should be cleared after an invalid submission"
        )

    def test_empty_submission_clears_stale_error(self):
        """Stale error from a previous failure must also be cleared."""
        state = _make_state(
            submitted="Teks lama.",
            error="🌐 **Layanan AI sementara tidak tersedia.**",
        )

        result_state = self._run_orchestrator_with_form_return((True, None), state)

        assert result_state[_KEY_ERROR] is None, (
            "Stale error should be cleared after an invalid submission"
        )

    def test_empty_submission_resets_submitted_key(self):
        """The last-submitted key must be reset so a future valid re-submission
        is not incorrectly treated as a duplicate."""
        state = _make_state(submitted="Teks lama.")

        result_state = self._run_orchestrator_with_form_return((True, None), state)

        assert result_state[_KEY_SUBMITTED] == "", (
            "_KEY_SUBMITTED should be reset to empty string on invalid submission"
        )

    def test_no_click_preserves_stale_result(self):
        """When the button was not clicked at all, existing state must be untouched."""
        from llm.schemas import Category, FeedbackAnalysis, Sentiment

        prior_result = FeedbackAnalysis(
            original_text="Produk sangat bagus!",
            sentiment=Sentiment.POSITIVE,
            category=Category.PRODUCT_QUALITY,
            key_issue="Kualitas produk sangat baik.",
            evidence="Produk",
            recommendation="Pertahankan standar kualitas.",
        )
        state = _make_state(submitted="Produk sangat bagus!", result=prior_result)

        result_state = self._run_orchestrator_with_form_return((False, None), state)

        assert result_state[_KEY_RESULT] is prior_result, (
            "Result must be preserved when no button was clicked"
        )
