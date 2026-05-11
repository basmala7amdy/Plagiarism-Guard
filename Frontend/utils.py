from __future__ import annotations
import math


def score_color(score: int) -> str:
    if score < 30:
        return "#00e5a0"  # green
    if score < 60:
        return "#f0c040"  # amber
    if score < 80:
        return "#ff8c42"  # orange
    return "#ff4d6d"      # red


def score_label(score: int) -> str:
    if score < 30:
        return "Low Risk"
    if score < 60:
        return "Moderate Risk"
    if score < 80:
        return "High Risk"
    return "Very High Risk"


def format_percentage(value: float | int, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}%"


def format_similarity(score: float) -> str:
    pct = score * 100
    if pct >= 80:
        level = "High"
    elif pct >= 50:
        level = "Medium"
    else:
        level = "Low"
    return f"{pct:.1f}% ({level})"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def word_count_label(count: int) -> str:
    if count >= 1000:
        return f"{count / 1000:.1f}k words"
    return f"{count} words"


def reading_time_minutes(word_count: int, wpm: int = 200) -> int:
    return max(1, math.ceil(word_count / wpm))  # minimum 1 minute


def truncate(text: str, max_chars: int = 120, ellipsis: str = "…") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


def preprocess_chart_data(chart_data: dict) -> dict:
    defaults = {
        "matched": 0,
        "original": 100,
        "source_breakdown": [],
        "similarity_timeline": [],
    }
    for key, default in defaults.items():
        chart_data.setdefault(key, default)

    chart_data["matched"] = clamp(chart_data["matched"], 0, 100)
    chart_data["original"] = clamp(chart_data["original"], 0, 100)
    return chart_data


def severity_badge_html(score: int) -> str:
    color = score_color(score)
    label = score_label(score)
    return (
        f'<span style="'
        f"background: {color}22; border: 1px solid {color}; color: {color}; "
        f"padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; "
        f'font-weight: 600;">'
        f"{label}</span>"
    )
