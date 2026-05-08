from __future__ import annotations
import streamlit as st
import time
from pathlib import Path

import api_client
import components
from utils import format_percentage, score_color

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlagiarismGuard — AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Helper: load CSS ──────────────────────────────────────────────────────────
def _load_css() -> None:
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css = f.read()
        except UnicodeDecodeError:
            # fallback (very safe)
            with open(css_path, "r", encoding="utf-8", errors="ignore") as f:
                css = f.read()

        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("styles.css not found. Visual styling may be incomplete.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🛡️ PlagiarismGuard")
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "AI-powered plagiarism detection that analyses your text "
            "against billions of online sources in real time."
        )
        st.markdown("---")
        st.markdown("### Settings")
        sensitivity = st.slider("Detection Sensitivity", 1, 10, 7)
        deep_scan = st.toggle("Deep Web Scan", value=True)
        include_paraphrase = st.toggle("Detect Paraphrasing", value=True)
        st.markdown("---")

        # Backend status indicator
        is_online = api_client.ping_backend()
        status_color = "#00e5a0" if is_online else "#ff4d6d"
        status_text  = "API Online" if is_online else "API Offline"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'font-size:0.82rem;color:{status_color};">'
            f'<div style="width:8px;height:8px;border-radius:50%;'
            f'background:{status_color};"></div>{status_text}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Backend: http://127.0.0.1:5000")

        st.session_state["sensitivity"] = sensitivity
        st.session_state["deep_scan"] = deep_scan
        st.session_state["include_paraphrase"] = include_paraphrase


# ── Hero section ──────────────────────────────────────────────────────────────
def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-logo">🛡️ PlagiarismGuard</div>
            <div class="hero-tagline">AI-Powered Content Authentication</div>
            <div class="hero-description">
                Instantly detect plagiarism, paraphrasing, and content theft
                across billions of web pages, academic papers, and publications.
            </div>
            <div class="hero-scan-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Input section ─────────────────────────────────────────────────────────────
def _render_input() -> tuple[str, bool]:
    st.markdown('<div class="section-header">📝 Submit Content for Analysis</div>', unsafe_allow_html=True)

    col_input, col_upload = st.columns([3, 2], gap="large")

    with col_input:
        user_text = st.text_area(
            label="Paste your text here",
            placeholder="Paste the text you want to check...",
            height=250,
            key="input_text",
            label_visibility="collapsed",
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf", "docx", "md"],
            label_visibility="collapsed",
            key="file_upload",
        )

        if uploaded_file is not None:
            try:
                file_text = uploaded_file.read().decode("utf-8", errors="ignore")
                if file_text.strip():
                    user_text = file_text
                    st.success(f"Loaded {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col = st.columns([1, 2, 1])[1]
    with btn_col:
        analyse_clicked = st.button(
            "🔍 ANALYZE",
            type="primary",
            use_container_width=True,
        )

    return user_text, analyse_clicked


# ── Results section ───────────────────────────────────────────────────────────
def _render_results(result: dict) -> None:
    st.markdown("<div class='results-wrapper'>", unsafe_allow_html=True)

    col_score, col_stats = st.columns([1, 2], gap="large")

    with col_score:
        components.render_score_card(result["score"])

    with col_stats:
        components.render_severity_badge(result["score"])
        components.render_stat_row(result)

    st.divider()

    tab_sources, tab_highlight, tab_charts = st.tabs(
        ["Sources", "Text", "Analytics"]
    )

    with tab_sources:
        components.render_sources(result.get("sources", []))

    with tab_highlight:
        components.render_highlighted_text(result.get("highlighted_text", ""))

    with tab_charts:
        components.render_charts(result.get("chart_data", {}))

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _load_css()
    _render_sidebar()
    _render_hero()

    st.divider()

    user_text, analyse_clicked = _render_input()

    if analyse_clicked:
        if not user_text or len(user_text.strip()) < 10:
            st.warning("Enter at least 10 characters.")
        else:
            with st.spinner("Scanning..."):
                try:
                    time.sleep(0.5)
                    result = api_client.check_plagiarism(user_text.strip())
                    st.session_state["last_result"] = result
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.session_state.pop("last_result", None)

    if "last_result" in st.session_state:
        st.divider()
        _render_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()