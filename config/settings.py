"""
config/settings.py — Application configuration loader.

This module is the single source of truth for every configurable value
in the application. It reads values from environment variables (loaded
from a .env file during local development) and validates them using
Pydantic so that the app fails fast with a clear error message when
something is misconfigured — rather than crashing deep inside a feature.

HOW IT WORKS
------------
1. pydantic-settings reads the .env file specified in SettingsConfigDict
(env_file=".env") — no explicit load_dotenv() call is needed.
2. Pydantic's BaseSettings reads those environment variables by name.
3. Validators on each field check types and acceptable ranges.
4. The module exposes a get_settings() helper that caches the Settings
object so the .env file is only parsed once per session.

WHY PYDANTIC?
-------------
Pydantic gives us automatic type coercion (the string "0.1" becomes the
float 0.1) and clear error messages when a required variable is missing,
all without writing manual validation code.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings, sourced from environment variables.

    Pydantic automatically reads each field from the environment variable
    whose name matches the field name in UPPER_CASE.
    Example: `llm_api_key` is read from the env var `LLM_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # extra="ignore" silently drops unknown env vars instead of raising.
        extra="ignore",
        # case_sensitive=False lets LLM_API_KEY match llm_api_key.
        case_sensitive=False,
    )

    # ── LLM Provider ──────────────────────────────────────────
    llm_api_key: str = Field(
        default="",
        description="API key for the LLM provider (Groq, OpenAI, etc.).",
    )

    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL of the OpenAI-compatible API endpoint.",
    )

    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Model identifier supported by the chosen provider.",
    )

    # ── LLM Inference Parameters ──────────────────────────────
    llm_max_tokens: int = Field(
        default=4096,
        ge=64,
        le=32768,
        description="Maximum tokens the model may generate per response.",
    )

    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. Keep low for consistent JSON output.",
    )

    llm_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Request timeout in seconds.",
    )

    # ── Retry & Rate-Limit ────────────────────────────────────
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retry attempts on transient API errors (429, 5xx).",
    )

    # ── Batch Processing ──────────────────────────────────────
    batch_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Rows processed per batch before pausing.",
    )

    batch_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="Seconds to pause between batches.",
    )

    # ── Upload Limits ─────────────────────────────────────────
    max_csv_size_mb: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum CSV upload size in megabytes.",
    )

    max_csv_rows: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of rows allowed in a CSV upload.",
    )

    # ── Derived / convenience properties ──────────────────────
    @property
    def max_csv_size_bytes(self) -> int:
        """Convert MB limit to bytes for direct comparison."""
        return self.max_csv_size_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        """
        True when a real API key has been provided.

        Rejects:
        - empty string
        - whitespace-only strings (e.g. "   ")
        - the .env.example placeholder value
        """
        stripped = self.llm_api_key.strip()
        return bool(stripped) and stripped != "your_api_key_here"

    # ── Validators ────────────────────────────────────────────
    @field_validator("llm_base_url")
    @classmethod
    def base_url_must_be_valid(cls, v: str) -> str:
        """
        Validate using urllib.parse.urlsplit so both scheme and netloc
        (hostname) are checked. Normalise by stripping trailing slashes.
        """
        v = v.rstrip("/")  # normalise — strip trailing slash
        parsed = urlsplit(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"LLM_BASE_URL must start with http:// or https://. Got: '{v}'"
            )
        if not parsed.netloc:
            raise ValueError(
                f"LLM_BASE_URL must contain a valid hostname. Got: '{v}'"
            )
        return v

    @field_validator("llm_model")
    @classmethod
    def model_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("LLM_MODEL must not be empty.")
        return v.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application Settings singleton.

    Using lru_cache means the .env file is parsed exactly once — the first
    time this function is called — and the same Settings object is reused
    everywhere. Call `get_settings.cache_clear()` in tests to reset state.
    """
    return Settings()
