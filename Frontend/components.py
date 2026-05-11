from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from utils import (
    score_color, score_label, format_percentage, format_similarity,
    severity_badge_html, preprocess_chart_data, word_count_label,
    reading_time_minutes, truncate,
)


def render_score_card(score: int) -> None:
    color = score_color(score)
    label = score_label(score)
    st.markdown(
        f"""
        <div class="score-card" style="border-color:{color};box-shadow:0 0 30px {color}44;">
            <div class="score-ring" style="color:{color};text-shadow:0 0 20px {color};">{score}%</div>
            <div class="score-label">{label}</div>
            <div class="score-sub">Overall Plagiarism Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_row(result: dict) -> None:
    col1, col2, col3 = st.columns(3)
    wc = result.get("word_count", 0)
    sc = result.get("sentence_count", 0)
    up = result.get("unique_phrases", 0)
    _mini_card(col1, "Words", word_count_label(wc), f"~{reading_time_minutes(wc)} min read")
    _mini_card(col2, "Sentences", str(sc), "detected in text")
    _mini_card(col3, "Unique Phrases", str(up), "not flagged")


def _mini_card(col, title: str, value: str, subtitle: str) -> None:
    with col:
        st.markdown(
            f'<div class="mini-card"><div class="mini-card-title">{title}</div>'
            f'<div class="mini-card-value">{value}</div>'
            f'<div class="mini-card-sub">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )


def render_sources(sources: list[dict]) -> None:
    st.markdown('<div class="section-header">Matched Sources</div>', unsafe_allow_html=True)
    if not sources:
        st.info("No matched sources found.")
        return
    for src in sources:
        _source_card(src)


def _source_card(src: dict) -> None:
    sim_pct = src.get("score", 0) * 100
    color = score_color(int(sim_pct))
    bar_width = int(sim_pct)  # CSS percentage width
    title = truncate(src.get("title", "Unknown source"), 60)
    domain = src.get("domain", "")
    url = src.get("url", "#")
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-header">
                <span class="source-title">{title}</span>
                <span class="source-badge" style="background:{color}22;color:{color};border-color:{color};">
                    {sim_pct:.1f}% match
                </span>
            </div>
            <div class="source-domain">{domain}</div>
            <div class="source-bar-bg">
                <div class="source-bar-fill" style="width:{bar_width}%;background:{color};"></div>
            </div>
            <a class="source-link" href="{url}" target="_blank">View Source &rarr;</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_highlighted_text(html_text: str) -> None:
    st.markdown('<div class="section-header">Text Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="highlight-container">'
        f'<div class="highlight-legend">'
        f'<span class="legend-dot plagiarized-dot"></span> Flagged as plagiarised &nbsp;&nbsp;'
        f'<span class="legend-dot original-dot"></span> Original content'
        f'</div><div class="highlight-body">{html_text}</div></div>',
        unsafe_allow_html=True,
    )


def render_charts(chart_data: dict) -> None:
    chart_data = preprocess_chart_data(chart_data)
    st.markdown('<div class="section-header">Analytics Dashboard</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        _donut_chart(chart_data)
    with col2:
        _source_bar_chart(chart_data)
    _timeline_chart(chart_data)


def _donut_chart(chart_data: dict) -> None:
    matched = chart_data["matched"]
    original = chart_data["original"]
    fig = go.Figure(
        go.Pie(
            labels=["Plagiarised", "Original"],
            values=[matched, original],
            hole=0.65,  # donut hole size
            marker=dict(colors=["#ff4d6d", "#00e5a0"], line=dict(color="#0a0e1a", width=3)),
            textinfo="percent",
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6f0", family="monospace"),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1, font=dict(size=12)),
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(text=f"<b>{matched}%</b>", x=0.5, y=0.5,
                          font=dict(size=28, color="#ff4d6d"), showarrow=False)],
    )
    st.plotly_chart(fig, use_container_width=True)


def _source_bar_chart(chart_data: dict) -> None:
    breakdown = chart_data.get("source_breakdown", [])
    if not breakdown:
        return
    labels = [item["label"] for item in breakdown]
    values = [item["value"] for item in breakdown]
    palette = ["#00b4ff", "#ff4d6d", "#f0c040", "#00e5a0", "#b060ff", "#ff8c42"]  # cycling neon colours
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values], textposition="outside",
            textfont=dict(color="#c8d6f0", size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6f0", family="monospace"),
        xaxis=dict(gridcolor="#1e2a44", ticksuffix="%",
                   range=[0, max(values) * 1.3 if values else 100]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=60), bargap=0.25,
    )
    st.plotly_chart(fig, use_container_width=True)


def _timeline_chart(chart_data: dict) -> None:
    timeline = chart_data.get("similarity_timeline", [])
    if not timeline:
        return
    segments = [item["segment"] for item in timeline]
    similarities = [item["similarity"] for item in timeline]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=segments, y=similarities, mode="lines+markers",
            line=dict(color="#00b4ff", width=3, shape="spline"),
            marker=dict(size=8, color="#00b4ff", line=dict(color="#0a0e1a", width=2)),
            fill="tozeroy", fillcolor="rgba(0,180,255,0.10)",  # shade area under line
            hovertemplate="<b>%{x}</b><br>Similarity: %{y}%<extra></extra>",
            name="Similarity",
        )
    )
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(240,192,64,0.4)",
                  annotation_text="50% threshold", annotation_font_color="#f0c040")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6f0", family="monospace"),
        xaxis=dict(gridcolor="#1e2a44"),
        yaxis=dict(gridcolor="#1e2a44", ticksuffix="%", range=[0, 105]),
        showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Similarity Timeline Across Document",
                   font=dict(size=14, color="#7fa8d4"), x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_severity_badge(score: int) -> None:
    st.markdown(severity_badge_html(score), unsafe_allow_html=True)
