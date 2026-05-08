"""
PlagiarismGuard - Utilities
===========================
Pure helper functions used by both components.py and ui.py.
No Streamlit imports here — keeps this module testable in isolation.
"""

from __future__ import annotations
import math


# ── Score colour thresholds ───────────────────────────────────────────────────
#  score 0–29   → green  (low risk)
#  score 30–59  → amber  (moderate)
#  score 60–79  → orange (high)
#  score 80–100 → red    (very high)

def score_color(score: int) -> str:
    """Return a CSS hex colour appropriate for the plagiarism score."""
    if score < 30:
        return "#00e5a0"   # neon green
    if score < 60:
        return "#f0c040"   # amber
    if score < 80:
        return "#ff8c42"   # orange
    return "#ff4d6d"       # red


def score_label(score: int) -> str:
    """Return a human-readable risk label for the score."""
    if score < 30:
        return "✅ Low Risk"
    if score < 60:
        return "⚠️ Moderate Risk"
    if score < 80:
        return "🔶 High Risk"
    return "🚨 Very High Risk"


def format_percentage(value: float | int, decimals: int = 1) -> str:
    """Format a 0-100 value as a percentage string, e.g. '87.0%'."""
    return f"{float(value):.{decimals}f}%"


def format_similarity(score: float) -> str:
    """
    Format a 0-1 similarity score as a percentage string with colour emoji.
    e.g. 0.91 → '91.0% 🔴'
    """
    pct = score * 100
    if pct >= 80:
        emoji = "🔴"
    elif pct >= 50:
        emoji = "🟡"
    else:
        emoji = "🟢"
    return f"{pct:.1f}% {emoji}"


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))


def word_count_label(count: int) -> str:
    """Return a friendly label for a word count, e.g. '1.2 k words'."""
    if count >= 1000:
        return f"{count / 1000:.1f}k words"
    return f"{count} words"


def reading_time_minutes(word_count: int, wpm: int = 200) -> int:
    """Estimate reading time in whole minutes."""
    return max(1, math.ceil(word_count / wpm))


def truncate(text: str, max_chars: int = 120, ellipsis: str = "…") -> str:
    """Truncate *text* to *max_chars*, appending *ellipsis* if needed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


def preprocess_chart_data(chart_data: dict) -> dict:
    """
    Validate and normalise the chart_data dict coming from the backend.
    Ensures all required keys are present with sensible defaults.
    """
    defaults = {
        "matched": 0,
        "original": 100,
        "source_breakdown": [],
        "similarity_timeline": [],
    }
    for key, default in defaults.items():
        chart_data.setdefault(key, default)

    # Clamp matched/original to valid range
    chart_data["matched"] = clamp(chart_data["matched"], 0, 100)
    chart_data["original"] = clamp(chart_data["original"], 0, 100)

    return chart_data


def severity_badge_html(score: int) -> str:
    """
    Return an inline HTML badge (<span>) coloured by severity.
    Rendered via st.markdown(..., unsafe_allow_html=True).
    """
    color = score_color(score)
    label = score_label(score)
    return (
        f'<span style="'
        f"background: {color}22; "
        f"border: 1px solid {color}; "
        f"color: {color}; "
        f"padding: 4px 14px; "
        f"border-radius: 20px; "
        f"font-size: 0.85rem; "
        f'font-weight: 600;">'
        f"{label}</span>"
    )
