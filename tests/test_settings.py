"""
tests/test_settings.py — Unit tests for config/settings.py.

These tests verify that the Settings class:
  - Loads and returns correct default values.
  - Correctly identifies when an API key is missing or is a placeholder.
  - Accepts valid custom values.
  - Rejects invalid values (bad URL, bad temperature range, etc.).
  - Computes derived properties correctly.

IMPORTANT: These tests do NOT require a real API key.
All tests override environment variables using monkeypatching so they
are completely self-contained and can run in any CI environment.

HOW TO RUN:
    pytest tests/test_settings.py -v
"""

import os

import pytest

# We import Settings directly (not get_settings) so we can instantiate
# a fresh object in each test without fighting the lru_cache.
from config.settings import Settings, get_settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_settings(**overrides) -> Settings:
    """
    Create a Settings instance with environment-variable overrides.

    Uses model_validate to bypass the lru_cache and produce a fresh instance,
    allowing each test to run in isolation.
    """
    # Build a minimal valid environment dict, then apply overrides.
    env = {
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "https://api.groq.com/openai/v1",
        "LLM_MODEL": "openai/gpt-oss-20b",
    }
    env.update({k.upper(): str(v) for k, v in overrides.items()})

    # Temporarily inject into os.environ so Pydantic's Settings picks them up.
    original = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v

    try:
        return Settings()
    finally:
        # Restore original environment (clean up after each test).
        for k, orig_v in original.items():
            if orig_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig_v


# ─────────────────────────────────────────────────────────────────────────────
# Default value tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsDefaults:
    """Verify that defaults match the documented .env.example values."""

    def test_default_base_url(self):
        s = make_settings()
        assert s.llm_base_url == "https://api.groq.com/openai/v1"

    def test_default_model(self):
        s = make_settings()
        assert s.llm_model == "openai/gpt-oss-20b"

    def test_default_max_tokens(self):
        s = make_settings()
        assert s.llm_max_tokens == 4096

    def test_default_temperature(self):
        s = make_settings()
        assert s.llm_temperature == 0.1

    def test_default_timeout(self):
        s = make_settings()
        assert s.llm_timeout_seconds == 30

    def test_default_max_retries(self):
        s = make_settings()
        assert s.llm_max_retries == 3

    def test_default_batch_size(self):
        s = make_settings()
        assert s.batch_size == 5

    def test_default_batch_delay(self):
        s = make_settings()
        assert s.batch_delay_seconds == 1.0

    def test_default_max_csv_size_mb(self):
        s = make_settings()
        assert s.max_csv_size_mb == 5

    def test_default_max_csv_rows(self):
        s = make_settings()
        assert s.max_csv_rows == 1000


# ─────────────────────────────────────────────────────────────────────────────
# API key status tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApiKeyStatus:
    """Verify has_api_key correctly identifies configured vs missing keys."""

    def test_empty_api_key_is_not_configured(self):
        s = make_settings(llm_api_key="")
        assert s.has_api_key is False

    def test_placeholder_api_key_is_not_configured(self):
        s = make_settings(llm_api_key="your_api_key_here")
        assert s.has_api_key is False

    def test_real_api_key_is_configured(self):
        s = make_settings(llm_api_key="gsk_abc123xyz")
        assert s.has_api_key is True

    def test_whitespace_only_api_key_is_not_configured(self):
        # has_api_key strips the value before checking, so whitespace-only
        # strings are treated the same as an empty string.
        s = make_settings(llm_api_key="   ")
        assert s.has_api_key is False


# ─────────────────────────────────────────────────────────────────────────────
# Derived property tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedProperties:
    """Verify computed properties return correct values."""

    def test_max_csv_size_bytes_default(self):
        s = make_settings()
        assert s.max_csv_size_bytes == 5 * 1024 * 1024

    def test_max_csv_size_bytes_custom(self):
        s = make_settings(max_csv_size_mb=10)
        assert s.max_csv_size_bytes == 10 * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# Custom override tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsOverrides:
    """Verify that custom env vars are applied correctly."""

    def test_custom_model_name(self):
        s = make_settings(llm_model="llama-3.3-70b-versatile")
        assert s.llm_model == "llama-3.3-70b-versatile"

    def test_custom_temperature(self):
        s = make_settings(llm_temperature=0.5)
        assert s.llm_temperature == 0.5

    def test_custom_batch_size(self):
        s = make_settings(batch_size=10)
        assert s.batch_size == 10

    def test_base_url_trailing_slash_stripped(self):
        """The validator should normalise trailing slashes."""
        s = make_settings(llm_base_url="https://api.groq.com/openai/v1/")
        assert not s.llm_base_url.endswith("/")

    def test_openai_base_url_accepted(self):
        s = make_settings(llm_base_url="https://api.openai.com/v1")
        assert s.llm_base_url == "https://api.openai.com/v1"

    def test_local_ollama_http_url_accepted(self):
        s = make_settings(llm_base_url="http://localhost:11434/v1")
        assert s.llm_base_url == "http://localhost:11434/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Validation error tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsValidation:
    """Verify that invalid values raise ValidationError with clear messages."""

    def test_invalid_base_url_scheme_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="LLM_BASE_URL must start with"):
            make_settings(llm_base_url="ftp://wrong.example.com")

    def test_empty_model_name_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="LLM_MODEL must not be empty"):
            make_settings(llm_model="   ")

    def test_temperature_above_max_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_settings(llm_temperature=1.5)

    def test_temperature_below_min_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_settings(llm_temperature=-0.1)

    def test_max_tokens_below_min_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_settings(llm_max_tokens=10)  # min is 64

    def test_batch_size_zero_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_settings(batch_size=0)  # min is 1

    def test_base_url_missing_hostname_raises(self):
        """urlsplit catches a URL with no netloc that startswith() would miss."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="valid hostname"):
            make_settings(llm_base_url="https://")


# ─────────────────────────────────────────────────────────────────────────────
# Cache / singleton tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSettingsSingleton:
    """Verify get_settings() returns a consistent singleton."""

    def test_get_settings_returns_settings_instance(self):
        get_settings.cache_clear()
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_returns_same_object_on_repeated_calls(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        # lru_cache should return the exact same object
        assert s1 is s2

    def teardown_method(self):
        """Reset the cache after each test in this class."""
        get_settings.cache_clear()
