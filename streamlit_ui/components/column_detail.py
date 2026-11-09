"""Modal dialog for detailed single-column analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding, normalize_missing_tokens
from streamlit_ui.components.column_actions import render_column_actions
from streamlit_ui.components.column_charts import (
    render_column_distribution_chart,
    render_target_relationship_analysis,
)
from streamlit_ui.components.common import highest_severity, render_severity
from streamlit_ui.components.ai_understanding import render_column_ai_understanding


@st.dialog("Detailed column analysis", width="large")
def render_column_detail_dialog(
    column: str,
    df: pd.DataFrame,
    target: str,
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    """Render a modal pop-up dialog for a single column with a 2-column layout."""
    series = df[column]
    column_findings = [f for f in findings if f.scope == column]
    issues = [f for f in column_findings if f.found]
    id_finding = next((f for f in issues if f.issue_type == "id_like"), None)
    severity = highest_severity(column_findings)
    applicable = {
        key: option
        for key, option in options.items()
        if option["column"] == column
    }
    relevance = (
        st.session_state.get("last_feature_relevance", {})
        if st.session_state.get("show_relevance_scores", False)
        else {}
    )
    relevance_suffix = (
        f" ({relevance[column]:.2%})" if column in relevance else ""
    )

    left_col, right_col = st.columns([1.65, 1.0], gap="large")

    with left_col:
        # 1. Column name
        st.subheader(f"{column}{relevance_suffix}")
        st.caption("Target column" if column == target else f"Type: {series.dtype}")

        # 2. Below it its chart
        st.markdown("**Column distribution**")
        render_column_distribution_chart(
            column, series, is_identifier=bool(id_finding)
        )

        # 3. Then information
        _render_column_information(series, bool(id_finding))

        # 4. Then dropdowns for its target analysis and ai understanding
        with st.expander("🎯 Target analysis", expanded=False):
            render_target_relationship_analysis(
                column,
                series,
                df,
                target,
                findings=findings,
                relevance_pct=relevance.get(column),
                is_identifier=bool(id_finding),
            )

        with st.expander("🤖 AI Understanding", expanded=False):
            render_column_ai_understanding(
                column,
                st.session_state.get("ai_understanding_result"),
            )

    with right_col:
        # Right side: another columnar section for all actions and apply button
        st.markdown("### Actions & Status")
        st.markdown(
            '<div class="dd-section-label">Severity</div>',
            unsafe_allow_html=True,
        )
        render_severity(severity)

        _render_column_problems(issues)

        st.markdown(
            '<div class="dd-section-label">Actions</div>',
            unsafe_allow_html=True,
        )
        if applicable:
            render_column_actions(
                column, series, applicable, key_prefix="dlg_"
            )
        elif column == target:
            st.markdown(
                '<div class="dd-card-actions">'
                '<div style="font-size:0.85rem; opacity:0.75; padding-top:0.4rem;">'
                'Target labels can be reviewed but not dropped.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="dd-card-actions">'
                '<div style="font-size:0.85rem; opacity:0.75; padding-top:0.4rem;">'
                'No action is currently needed.</div></div>',
                unsafe_allow_html=True,
            )


def _render_column_information(
    series: pd.Series,
    is_identifier: bool,
) -> None:
    st.markdown(
        '<div class="dd-section-label">Information</div>',
        unsafe_allow_html=True,
    )
    observed = normalize_missing_tokens(series)
    info = [
        f"Missing: {observed.isna().sum():,} ({observed.isna().mean():.1%})",
        f"Unique: {observed.nunique(dropna=True):,}",
    ]
    if is_identifier:
        info.append(
            "Identifier warning: almost every non-missing value is "
            "unique; numeric summaries are not meaningful."
        )
    elif (
        pd.api.types.is_numeric_dtype(series)
        and series.nunique(dropna=True) > 2
    ):
        info.extend(
            [
                f"Median: {series.median():,.2f}",
                f"Range: {series.min():,.2f} to {series.max():,.2f}",
            ]
        )
    st.markdown(
        f'<div class="dd-card-info">{"<br>".join(info)}</div>',
        unsafe_allow_html=True,
    )


def _render_column_problems(issues: list[Finding]) -> None:
    st.markdown(
        '<div class="dd-section-label">Problems</div>',
        unsafe_allow_html=True,
    )
    if not issues:
        st.markdown(
            '<div class="dd-card-problems">'
            '<div style="color:var(--dd-green); background:var(--dd-green-soft); '
            'padding:0.35rem 0.6rem; border-radius:0.45rem; font-size:0.82rem; font-weight:600; '
            'border:1px solid rgba(22,132,91,0.2);">'
            '✓ No issue found</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    items_html = []
    for finding in issues:
        title = (
            "Needs manual review"
            if finding.issue_type == "manual_review"
            else finding.issue_type.replace("_", " ").title()
        )
        reason = finding.reason
        items_html.append(
            f'<div style="margin-bottom:0.3rem;">'
            f'<div style="font-weight:650; font-size:0.85rem;">• {title}</div>'
            f'<div style="font-size:0.78rem; opacity:0.75; line-height:1.25;">{reason}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="dd-card-problems">{"".join(items_html)}</div>',
        unsafe_allow_html=True,
    )

