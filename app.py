"""
app.py — Streamlit application entry point.

Run the app with:
    streamlit run app.py

HOW STREAMLIT WORKS
────────────────────
Streamlit re-runs this entire file top-to-bottom on every user interaction.
All persistent state is stored in `st.session_state` — it survives re-runs
unlike local variables.

ARCHITECTURE
────────────
  Presentation : app.py  +  ui/manual_input.py  +  ui/batch_input.py
  Business     : core/analyzer.py  +  core/batch_analyzer.py
  AI / network : llm/service.py  (never imported here)
"""

import streamlit as st

from config.settings import get_settings

# ── Page configuration ────────────────────────────────────────────────────────
# Must be the FIRST Streamlit command called in the script.
st.set_page_config(
    page_title="Customer Feedback Insight Agent",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load settings ─────────────────────────────────────────────────────────────
settings = get_settings()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    st.divider()

    st.markdown(f"**Endpoint:** `{settings.llm_base_url}`")
    st.markdown(f"**Model:** `{settings.llm_model}`")
    st.markdown(f"**Maks token:** `{settings.llm_max_tokens}`")
    st.markdown(f"**Temperatur:** `{settings.llm_temperature}`")
    st.divider()

    if settings.has_api_key:
        st.success("✅ API key terkonfigurasi")
    else:
        st.warning(
            "⚠️ **API key belum diatur.**\n\n"
            "Salin `.env.example` ke `.env` dan isi `LLM_API_KEY` "
            "dengan API key Groq Anda untuk memulai."
        )

    st.divider()
    st.caption("Customer Feedback Insight Agent · M5")


# ── Main header ───────────────────────────────────────────────────────────────
st.title("💬 Customer Feedback Insight Agent")
st.markdown(
    "**Sistem Analisis Sentimen & Rekomendasi Berbasis AI**\n\n"
    "Analisis ulasan pelanggan secara individual atau batch menggunakan AI — "
    "dalam Bahasa Indonesia maupun Bahasa Inggris."
)
st.divider()

# ── API key guard ─────────────────────────────────────────────────────────────
if not settings.has_api_key:
    st.error(
        "🔑 **API key belum dikonfigurasi.**\n\n"
        "Tambahkan `LLM_API_KEY` ke file `.env` Anda, lalu muat ulang halaman ini.\n\n"
        "Dapatkan API key gratis di [console.groq.com](https://console.groq.com).",
        icon="🔑",
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_manual, tab_batch = st.tabs(["📝 Analisis Manual", "📂 Analisis Batch CSV"])

with tab_manual:
    from ui.manual_input import render_manual_input_section  # noqa: E402
    render_manual_input_section()

with tab_batch:
    from ui.batch_input import render_batch_input_section  # noqa: E402
    render_batch_input_section()
