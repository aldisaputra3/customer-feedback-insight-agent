"""
ui/manual_input.py — Manual feedback input form and analysis result display.

Responsibilities
────────────────
• Render the text input form and Analyze button.
• Validate client-side constraints (empty, length limit).
• Call core.analyzer.analyze_feedback via a Streamlit spinner.
• Cache results in session_state to avoid repeat API calls.
• Render the structured analysis result card.
• Render all service error states with actionable, non-technical messages.

This module must NOT import the OpenAI SDK directly.
All AI calls are delegated to core.analyzer.
"""

from __future__ import annotations

import streamlit as st

from core.analyzer import analyze_feedback
from llm.schemas import FeedbackAnalysis
from llm.service import (
    AnalysisError,
    AuthenticationError,
    InvalidRequestError,
    ResponseParseError,
    TransientError,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_INPUT_CHARS: int = 2000

_PLACEHOLDER = (
    "Contoh: Produk yang saya terima rusak dan pengirimannya sangat lambat. "
    "Layanan pelanggan tidak membantu sama sekali.\n\n"
    "Example: The item arrived damaged and delivery took two weeks. "
    "Customer support was unresponsive."
)

_SENTIMENT_CONFIG: dict[str, dict[str, str]] = {
    "Positive": {"label": "😊 Positif",   "color": "#16a34a", "bg": "#f0fdf4"},
    "Negative": {"label": "😞 Negatif",   "color": "#dc2626", "bg": "#fef2f2"},
    "Neutral":  {"label": "😐 Netral",    "color": "#6b7280", "bg": "#f9fafb"},
}

# ── Session-state keys ────────────────────────────────────────────────────────

_KEY_SUBMITTED  = "manual_last_submitted"
_KEY_RESULT     = "manual_analysis_result"
_KEY_ERROR      = "manual_analysis_error"


def _init_state() -> None:
    """Initialise session-state keys on first run."""
    for key, default in [
        (_KEY_SUBMITTED, ""),
        (_KEY_RESULT,    None),
        (_KEY_ERROR,     None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ── Input form ────────────────────────────────────────────────────────────────

def _render_input_form() -> tuple[bool, str | None]:
    """
    Render the text area and submit button.

    Returns a ``(button_clicked, valid_text)`` tuple:

    * ``(False, None)``  — button was not clicked this run.
    * ``(True,  None)``  — button was clicked but input failed validation.
    * ``(True,  text)``  — button was clicked and input is valid; ``text`` is
                           the stripped feedback ready for analysis.

    Separating "clicked" from "valid" lets the caller clear stale session
    state when the user deliberately submits an invalid (e.g. empty) form,
    so old results from a previous run are never shown alongside a
    validation error.
    """
    st.subheader("📝 Masukkan Ulasan Pelanggan")
    st.caption(
        "Masukkan satu ulasan pelanggan dalam Bahasa Indonesia atau Bahasa Inggris. "
        "AI akan menganalisis sentimen, kategori, dan memberikan rekomendasi."
    )

    feedback_text: str = st.text_area(
        label="Teks ulasan",
        placeholder=_PLACEHOLDER,
        height=150,
        max_chars=MAX_INPUT_CHARS,
        label_visibility="collapsed",
        key="manual_input_textarea",
    )

    char_count = len(feedback_text)
    col_btn, col_count = st.columns([3, 1])

    with col_count:
        color = "#dc2626" if char_count >= MAX_INPUT_CHARS else "#6b7280"
        st.markdown(
            f"<p style='text-align:right; color:{color}; font-size:0.8rem; "
            f"margin-top:0.4rem;'>{char_count} / {MAX_INPUT_CHARS}</p>",
            unsafe_allow_html=True,
        )

    with col_btn:
        submitted = st.button(
            "🔍 Analisis Ulasan",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return False, None

    # ── Client-side validation ────────────────────────────────────────────────
    if not feedback_text or not feedback_text.strip():
        st.error("⚠️ Kolom ulasan tidak boleh kosong. Silakan masukkan teks ulasan.", icon="⚠️")
        return True, None

    if len(feedback_text) > MAX_INPUT_CHARS:
        st.error(
            f"⚠️ Ulasan terlalu panjang ({len(feedback_text)} karakter). "
            f"Batas maksimum adalah {MAX_INPUT_CHARS} karakter.",
            icon="⚠️",
        )
        return True, None

    return True, feedback_text.strip()


# ── Analysis runner ───────────────────────────────────────────────────────────

def _run_analysis(feedback_text: str) -> None:
    """
    Call the analyzer and store the result or error in session_state.

    Cache hit (skip API call): same text AND a successful result already exists.
    Cache miss: same text but the previous attempt failed — allow retry.
    Cache miss: different text — always call the API.
    """
    if (
        feedback_text == st.session_state[_KEY_SUBMITTED]
        and st.session_state[_KEY_RESULT] is not None
    ):
        # Same text, prior call succeeded — reuse the cached result.
        return

    # Reset previous state before the new call.
    st.session_state[_KEY_SUBMITTED] = feedback_text
    st.session_state[_KEY_RESULT] = None
    st.session_state[_KEY_ERROR] = None

    with st.spinner("⏳ Menganalisis ulasan…"):
        try:
            result = analyze_feedback(feedback_text)
            st.session_state[_KEY_RESULT] = result
        except AuthenticationError:
            st.session_state[_KEY_ERROR] = (
                "🔑 **Autentikasi gagal.** "
                "API key tidak ditemukan, tidak valid, atau telah dicabut. "
                "Periksa nilai `LLM_API_KEY` di file `.env` Anda."
            )
        except TransientError:
            st.session_state[_KEY_ERROR] = (
                "🌐 **Layanan AI sementara tidak tersedia.** "
                "Terjadi kesalahan jaringan atau batas permintaan (rate limit) telah tercapai. "
                "Tunggu beberapa detik lalu coba kembali."
            )
        except InvalidRequestError:
            st.session_state[_KEY_ERROR] = (
                "⚙️ **Konfigurasi tidak valid.** "
                "Model atau parameter yang dikonfigurasi tidak dikenali oleh penyedia API. "
                "Periksa nilai `LLM_MODEL` di file `.env` Anda."
            )
        except ResponseParseError as exc:
            # Surface actionable detail (e.g. token limit hit) without leaking internals.
            msg = str(exc)
            if "max_completion_tokens" in msg or "LLM_MAX_TOKENS" in msg:
                st.session_state[_KEY_ERROR] = (
                    "📏 **Respons terpotong.** "
                    "Model berhenti sebelum menyelesaikan JSON karena batas token tercapai. "
                    "Naikkan nilai `LLM_MAX_TOKENS` di file `.env` Anda."
                )
            else:
                st.session_state[_KEY_ERROR] = (
                    "🤖 **Format respons tidak valid.** "
                    "Model mengembalikan data yang tidak dapat diparsing. "
                    "Coba kirimkan ulasan sekali lagi."
                )
        except AnalysisError:
            st.session_state[_KEY_ERROR] = (
                "❌ **Terjadi kesalahan tidak terduga.** "
                "Silakan coba lagi. Jika masalah berlanjut, periksa konfigurasi Anda."
            )
        except ValueError:
            st.session_state[_KEY_ERROR] = (
                "⚠️ **Teks ulasan tidak boleh kosong.**"
            )


# ── Result display ────────────────────────────────────────────────────────────

def _render_result(result: FeedbackAnalysis) -> None:
    """Render the structured analysis result card."""
    st.divider()
    st.subheader("✅ Hasil Analisis")

    sentiment_key = result.sentiment.value  # "Positive" | "Negative" | "Neutral"
    cfg = _SENTIMENT_CONFIG.get(sentiment_key, _SENTIMENT_CONFIG["Neutral"])

    # ── Sentiment badge ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:{cfg['bg']};
            color:{cfg['color']};
            border:1.5px solid {cfg['color']};
            border-radius:8px;
            padding:6px 18px;
            font-size:1.1rem;
            font-weight:700;
            margin-bottom:1rem;
        ">{cfg['label']}</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Category + Key Issue (side by side) ───────────────────────────────────
    col_cat, col_issue = st.columns(2)
    with col_cat:
        st.markdown("**🏷️ Kategori**")
        st.markdown(
            f"<span style='font-size:1rem; font-weight:600;'>{result.category.value}</span>",
            unsafe_allow_html=True,
        )
    with col_issue:
        st.markdown("**💡 Isu Utama**")
        st.markdown(result.key_issue)

    st.divider()

    # ── Evidence ──────────────────────────────────────────────────────────────
    st.markdown("**🔍 Bukti dari Ulasan**")
    st.markdown(
        f"> *\"{result.evidence}\"*",
    )

    # ── Recommendation ────────────────────────────────────────────────────────
    st.markdown("**🎯 Rekomendasi**")
    st.info(result.recommendation, icon="💡")

    # ── Original text (collapsible) ───────────────────────────────────────────
    with st.expander("📄 Lihat teks ulasan asli"):
        st.text(result.original_text)


# ── Error display ─────────────────────────────────────────────────────────────

def _render_error(message: str) -> None:
    st.divider()
    st.error(message, icon="❌")


# ── Public entry point ────────────────────────────────────────────────────────

def render_manual_input_section() -> None:
    """
    Top-level function called by app.py.

    Orchestrates the form, analysis call, and result/error display.

    State transitions on button click:
      - Valid input   → call API, show result or error.
      - Invalid input → clear stale result/error, show only the validation
                        message.  This prevents old results from a prior run
                        from appearing alongside a fresh validation error.
      - No click      → display whatever is already in session_state
                        (supports page-refresh / navigation without re-calling).
    """
    _init_state()

    button_clicked, submitted_text = _render_input_form()

    if button_clicked and submitted_text is not None:
        # Valid submission — run (or cache-hit) analysis.
        _run_analysis(submitted_text)
    elif button_clicked and submitted_text is None:
        # Button was clicked but input was invalid — clear stale state so the
        # old result is not displayed alongside the validation error message.
        st.session_state[_KEY_RESULT] = None
        st.session_state[_KEY_ERROR] = None
        st.session_state[_KEY_SUBMITTED] = ""

    # Display persisted state (result or error from this or the previous run).
    if st.session_state[_KEY_RESULT] is not None:
        _render_result(st.session_state[_KEY_RESULT])
    elif st.session_state[_KEY_ERROR] is not None:
        _render_error(st.session_state[_KEY_ERROR])
