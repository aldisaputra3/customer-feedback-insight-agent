"""
llm/client.py — Factory for the OpenAI-compatible API client.

Responsibilities
────────────────
• Read connection settings from the application Settings object.
• Construct and return an `openai.OpenAI` client configured for the
  Groq-compatible base URL.
• Keep client construction injectable: callers can supply a pre-built
  client for testing instead of creating a real one.

IMPORTANT: This module must never log or expose the API key.
"""

from __future__ import annotations

import openai

from config.settings import Settings, get_settings


def make_client(settings: Settings | None = None) -> openai.OpenAI:
    """
    Build and return an OpenAI-compatible API client.

    The client is configured using the application Settings:
      - ``api_key``      — read from Settings.llm_api_key
      - ``base_url``     — read from Settings.llm_base_url
      - ``timeout``      — read from Settings.llm_timeout_seconds
      - ``max_retries``  — set to 0 because llm/service.py manages retries
                           manually with exponential backoff

    Args:
        settings: Optional Settings instance.  Defaults to get_settings().
                  Pass a custom Settings object in tests to avoid reading .env.

    Returns:
        A configured ``openai.OpenAI`` instance.

    Note:
        The returned client is NOT cached.  Callers that need a singleton
        should cache it themselves (e.g. via functools.lru_cache or
        st.cache_resource in Streamlit).
    """
    if settings is None:
        settings = get_settings()

    # api_key may be an empty string when LLM_API_KEY is not configured.
    # The SDK accepts an empty string; the first API call will then fail with
    # AuthenticationError, which llm/service.py maps to our AuthenticationError.
    # We never log or expose the key value here.
    return openai.OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        # timeout drives the per-request deadline; service.py controls retries.
        timeout=float(settings.llm_timeout_seconds),
        # Disable the SDK's built-in retry so service.py centralises the policy.
        max_retries=0,
    )
