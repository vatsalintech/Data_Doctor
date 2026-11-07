"""Chart helpers for per-column visualisations (distribution + target analysis)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding
from streamlit_ui.visualizations.categorical_charts import (
    compact_frequency_chart,
    detail_frequency_chart,
)
from streamlit_ui.visualizations.numeric_charts import (
    compact_histogram,
    detail_histogram,
)
from streamlit_ui.visualizations.target_analysis import (
    assess_feature_target_relationship,
    categorical_target_distribution,
    numeric_target_distribution,
)


def render_compact_chart(series: pd.Series) -> None:
    """Render a small inline chart preview inside the column card."""
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            chart = compact_histogram(series)
        else:
            chart = compact_frequency_chart(series)
        st.altair_chart(chart, width="stretch")
    except Exception:
        # Never let a chart error break the card
        pass


def render_column_distribution_chart(
    column: str,
    series: pd.Series,
    is_identifier: bool = False,
) -> None:
    """Render full-size distribution chart for the column."""
    if is_identifier:
        st.caption("Identifier column — distribution chart omitted.")
        return

    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            st.altair_chart(
                detail_histogram(series, column_name=column),
                width="stretch",
            )
        else:
            st.altair_chart(
                detail_frequency_chart(series, column_name=column),
                width="stretch",
            )
    except Exception:
        st.caption("Distribution chart unavailable for this column.")


def render_target_relationship_analysis(
    column: str,
    series: pd.Series,
    df: pd.DataFrame,
    target: str,
    findings: list[Finding] | None = None,
    relevance_pct: float | None = None,
    is_identifier: bool = False,
) -> None:
    """Render feature necessity assessment, target relationship, and target-distribution chart."""
    if target not in df.columns or column == target:
        if column == target:
            st.caption("This is the target column to be predicted.")
        else:
            st.caption("Select a target column to see feature-vs-target analysis.")
        return

    target_series = df[target]

    # Feature Necessity & Target Relationship
    assessment = assess_feature_target_relationship(
        feature=series,
        target=target_series,
        findings=findings or [],
        feature_name=column,
        target_name=target,
        relevance_pct=relevance_pct,
    )
    st.markdown("**Target Relationship & Necessity**")
    badge_color = assessment.get("badge_color", "#16845b")
    status = assessment.get("status", "Analyzed")
    necessity = assessment.get("is_necessary", "Useful")
    reason = assessment.get("reason", "")
    badge_html = (
        f'<div style="margin-bottom:0.4rem;">'
        f'<span style="background-color:{badge_color}22; color:{badge_color}; '
        f'border:1px solid {badge_color}55; padding:3px 10px; border-radius:12px; '
        f'font-weight:700; font-size:0.8rem;">'
        f'{status} • {necessity}</span>'
        f'</div>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.caption(reason)

    if is_identifier:
        return

    st.markdown(f"**{column} × {target} distribution**")
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            chart = numeric_target_distribution(
                series,
                target_series,
                feature_name=column,
                target_name=target,
            )
        else:
            chart = categorical_target_distribution(
                series,
                target_series,
                feature_name=column,
                target_name=target,
            )
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        else:
            st.caption("Not enough data to show a target distribution chart.")
    except Exception:
        st.caption("Target distribution chart unavailable for this column.")

