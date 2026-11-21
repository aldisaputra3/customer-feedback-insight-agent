"""
scripts/test_llm.py — Manual integration test for the LLM analysis pipeline.

PURPOSE
───────
Verify that the full stack (settings → client → prompts → model → validation)
works correctly when you have a real API key configured in your .env file.

This script is NOT part of the automated pytest suite.
Run it manually after setting LLM_API_KEY in your .env.

HOW TO RUN
──────────
  # From the project root directory:
  python scripts/test_llm.py

WHAT IT DOES
────────────
1. Loads settings from .env (shows whether API key is configured).
2. Sends one Indonesian sample review to the configured LLM.
3. Prints the validated structured result as formatted JSON.
4. Never prints the API key — only its presence is reported.

REQUIREMENTS
────────────
  - LLM_API_KEY must be set in .env (copy from .env.example if needed).
  - Run from the project root so imports resolve correctly.
"""

from __future__ import annotations

import sys

# Ensure the project root is on sys.path when the script is run directly.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from core.analyzer import analyze_feedback
from llm.service import (
    AnalysisError,
    AuthenticationError,
    InvalidRequestError,
    ResponseParseError,
    TransientError,
)

# ── Sample review (Indonesian) ────────────────────────────────────────────────

SAMPLE_REVIEW = (
    "Aplikasinya mudah digunakan dan tampilannya modern. "
    "Tapi pengiriman barang saya terlambat 5 hari dari jadwal "
    "dan tidak ada notifikasi apapun dari penjual. "
    "Produknya sendiri bagus dan sesuai deskripsi."
)


def main() -> None:
    settings = get_settings()

    # ── Pre-flight check ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Customer Feedback Insight Agent — LLM Integration Test")
    print("=" * 60)
    print(f"  Base URL : {settings.llm_base_url}")
    print(f"  Model    : {settings.llm_model}")
    print(f"  API Key  : {'✅ configured' if settings.has_api_key else '❌ NOT set'}")
    print()

    if not settings.has_api_key:
        print("ERROR: LLM_API_KEY is not configured.")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API key to .env:  LLM_API_KEY=<your_key>")
        sys.exit(1)

    # ── Run analysis ──────────────────────────────────────────────────────────
    print("Sample review:")
    print(f"  {SAMPLE_REVIEW!r}")
    print()
    print("Sending to LLM …")

    try:
        result = analyze_feedback(SAMPLE_REVIEW)
    except AuthenticationError as exc:
        print(f"\n❌ Authentication error: {exc}")
        print("   Check that LLM_API_KEY in your .env is valid.")
        sys.exit(2)
    except InvalidRequestError as exc:
        print(f"\n❌ Invalid request: {exc}")
        print("   Check LLM_MODEL and LLM_BASE_URL in your .env.")
        sys.exit(3)
    except TransientError as exc:
        print(f"\n❌ Transient error (retries exhausted): {exc}")
        sys.exit(4)
    except ResponseParseError as exc:
        print(f"\n❌ Response parse error: {exc}")
        sys.exit(5)
    except AnalysisError as exc:
        print(f"\n❌ Analysis error: {exc}")
        sys.exit(6)

    # ── Display result ────────────────────────────────────────────────────────
    print("\n✅ Analysis complete!\n")
    print(result.model_dump_json(indent=2))
    print()


if __name__ == "__main__":
    main()
