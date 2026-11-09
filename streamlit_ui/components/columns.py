"""Per-column analysis card grid with pagination and action feedback."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding, normalize_missing_tokens
from streamlit_ui.components.ai_understanding import render_column_ai_understanding
from streamlit_ui.components.column_actions import render_column_actions
from streamlit_ui.components.column_charts import (
    render_column_distribution_chart,
    render_target_relationship_analysis,
)
from streamlit_ui.components.column_detail import render_column_detail_dialog
from streamlit_ui.components.common import highest_severity, render_severity
from streamlit_ui.config import SEVERITY_ORDER


PAGE_SIZE = 12

# Issue types that make other changes largely irrelevant until resolved
_PRIORITY_ISSUE_TYPES = {"id_like", "target_copy", "suspicious_predictor"}


# ─────────────────────────────────────────────────────────────────────────────
# Action feedback toast
# ─────────────────────────────────────────────────────────────────────────────


def render_action_feedback() -> None:
    """Display a success or error toast for the most recent action, then clear it."""
    success_msg = st.session_state.pop("_last_action_success", None)
    error_msg = st.session_state.pop("_last_action_error", None)
    if success_msg:
        st.toast(success_msg, icon="✅")
    if error_msg:
        st.toast(error_msg, icon="❌")


# ─────────────────────────────────────────────────────────────────────────────
# Priority-drop warning banner
# ─────────────────────────────────────────────────────────────────────────────


def render_manual_review_alert(findings: list[Finding]) -> None:
    """Warn that some labels were left unchanged on purpose."""
    reviews = [
        finding
        for finding in findings
        if finding.found and finding.issue_type == "manual_review"
    ]
    if not reviews:
        return
    names = [
        "column names"
        if finding.scope == "dataset"
        else str(finding.scope)
        for finding in reviews
    ]
    shown = names[:12]
    extra = len(names) - len(shown)
    suffix = f", and {extra} more" if extra else ""
    listed = ", ".join(f"**{name}**" for name in shown)
    st.warning(
        "Some values need a manual look. Data Doctor will not guess which "
        "spellings or stray labels belong together, because merging the "
        "wrong ones would change what the data means.\n\n"
        f"Review these yourself: {listed}{suffix}.\n\n"
        "Each flagged column says what to look at. Nothing is changed "
        "until you edit the file or apply an action you choose."
    )


def render_priority_drop_warning(findings: list[Finding]) -> None:
    """Show a banner when high-priority columns (ID / leakage) need to be dropped first.

    Changes to other columns are largely irrelevant until these are dropped,
    because they can dominate or completely mask the model's behaviour.
    """
    priority_cols: list[tuple[str, str]] = []  # (column, human-readable reason)
    for finding in findings:
        if finding.found and finding.issue_type in _PRIORITY_ISSUE_TYPES:
            label = {
                "id_like": "ID-like column (almost all unique values)",
                "target_copy": "Target copy / direct leakage",
                "suspicious_predictor": "Suspicious near-perfect predictor (possible leakage)",
            }.get(finding.issue_type, finding.issue_type.replace("_", " ").title())
            priority_cols.append((finding.scope, label))

    if not priority_cols:
        return

    col_lines = "\n".join(
        f"- **{col}** — {reason}" for col, reason in priority_cols
    )
    st.error(
        "🚨 **Action required before other changes will be meaningful.**\n\n"
        "The following column(s) are flagged as high-priority and should be "
        "**dropped first**. Until they are removed, changes to other columns "
        "(imputation, outlier capping, encoding, etc.) may have little or no "
        "visible effect on model metrics — these columns effectively override "
        "everything else:\n\n"
        + col_lines
        + "\n\n"
        "👉 Find each column's card below and choose **Drop column** to remove it.",
        icon="🚨",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Column grid
# ─────────────────────────────────────────────────────────────────────────────


def render_column_grid(
    df: pd.DataFrame,
    target: str,
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    # Show action feedback first so it appears at the top of the section
    render_action_feedback()

    title_column, toggle_column = st.columns([4, 1], vertical_alignment="center")
    with title_column:
        st.subheader("Column analysis")
    with toggle_column:
        st.toggle(
            "Relevance score",
            key="show_relevance_scores",
            disabled=not bool(
                st.session_state.get("last_feature_relevance")
            ),
            help="Show scores from the most recent model training.",
        )
    st.caption(
        "Each card keeps the column facts, detected problems, severity, "
        "and available actions together. Expand 'Column chart distribution' or 'Target analysis' for charts and feature necessity."
    )
    original_order = {
        column: index for index, column in enumerate(df.columns)
    }
    severity_by_column = {
        column: highest_severity(
            [
                finding
                for finding in findings
                if finding.scope == column
            ]
        )
        for column in df.columns
    }
    column_names = sorted(
        df.columns,
        key=lambda column: (
            -SEVERITY_ORDER[severity_by_column[column]],
            original_order[column],
        ),
    )
    total_columns = len(column_names)
    page_count = max(1, (total_columns + PAGE_SIZE - 1) // PAGE_SIZE)
    current_page = st.session_state.get("column_page", 1)
    if current_page > page_count:
        current_page = page_count
    if current_page < 1:
        current_page = 1
    st.session_state.column_page = current_page

    visible_columns = column_names[
        (current_page - 1) * PAGE_SIZE : current_page * PAGE_SIZE
    ]
    for start in range(0, len(visible_columns), 4):
        row = st.columns(4)
        for container, column in zip(
            row, visible_columns[start : start + 4]
        ):
            with container:
                _render_column_card(
                    column,
                    df,
                    target,
                    findings,
                    options,
                )

    _render_pagination(current_page, page_count, total_columns)


# ─────────────────────────────────────────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────────────────────────────────────────


def _render_pagination(
    current_page: int, page_count: int, total_columns: int
) -> None:
    if page_count <= 1:
        return

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    if page_count <= 7:
        pages = list(range(1, page_count + 1))
    else:
        left = max(1, current_page - 2)
        right = min(page_count, current_page + 2)
        pages = list(range(left, right + 1))

    btn_count = len(pages) + 2
    spacer_width = max(1.0, (14.0 - btn_count) / 2.0)
    col_specs = [spacer_width] + [1.0] * btn_count + [spacer_width]
    cols = st.columns(col_specs, vertical_alignment="center")

    with cols[1]:
        if st.button(
            "‹",
            key="col_page_prev",
            disabled=(current_page <= 1),
            width="stretch",
            help="Previous page",
        ):
            st.session_state.column_page = max(1, current_page - 1)
            st.rerun()

    for idx, p in enumerate(pages):
        with cols[2 + idx]:
            is_active = (p == current_page)
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                str(p),
                key=f"col_page_{p}",
                type=btn_type,
                width="stretch",
                help=f"Page {p}",
            ):
                if not is_active:
                    st.session_state.column_page = p
                    st.rerun()

    with cols[2 + len(pages)]:
        if st.button(
            "›",
            key="col_page_next",
            disabled=(current_page >= page_count),
            width="stretch",
            help="Next page",
        ):
            st.session_state.column_page = min(page_count, current_page + 1)
            st.rerun()

    start_idx = (current_page - 1) * PAGE_SIZE + 1
    end_idx = min(total_columns, current_page * PAGE_SIZE)
    st.markdown(
        f"<div style='text-align: center; color: var(--text-color); opacity: 0.65; "
        f"font-size: 0.82rem; margin-top: 0.35rem;'>"
        f"Showing {start_idx}–{end_idx} of {total_columns} columns &bull; Page {current_page} of {page_count}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single column card
# ─────────────────────────────────────────────────────────────────────────────


def _render_column_card(
    column: str,
    df: pd.DataFrame,
    target: str,
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    series = df[column]
    column_findings = [
        finding for finding in findings if finding.scope == column
    ]
    issues = [finding for finding in column_findings if finding.found]
    id_finding = next(
        (
            finding
            for finding in issues
            if finding.issue_type == "id_like"
        ),
        None,
    )
    severity = highest_severity(column_findings)
    applicable = {
        key: option
        for key, option in options.items()
        if option["column"] == column
    }

    with st.container(border=True):
        relevance = (
            st.session_state.get("last_feature_relevance", {})
            if st.session_state.get("show_relevance_scores", False)
            else {}
        )
        relevance_suffix = (
            f" ({relevance[column]:.2%})" if column in relevance else ""
        )
        hdr_title, hdr_expand = st.columns([5, 1], vertical_alignment="center")
        with hdr_title:
            st.markdown(f"#### {column}{relevance_suffix}")
        with hdr_expand:
            if st.button(
                "⛶",
                key=f"expand_col_{column}",
                help=f"Open detailed analysis pop up for {column}",
                width="stretch",
            ):
                render_column_detail_dialog(
                    column, df, target, findings, options
                )

        st.caption("Target column" if column == target else str(series.dtype))
        _render_column_information(series, bool(id_finding))

        _render_column_problems(issues)
        st.markdown(
            '<div class="dd-section-label">Severity</div>',
            unsafe_allow_html=True,
        )
        render_severity(severity)
        st.markdown(
            '<div class="dd-section-label">Actions</div>',
            unsafe_allow_html=True,
        )
        if applicable:
            render_column_actions(column, series, applicable)
        elif column == target:
            st.markdown(
                '<div class="dd-card-actions">'
                '<div style="font-size:0.82rem; opacity:0.7; padding-top:0.4rem;">'
                'Target labels can be reviewed but not dropped.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="dd-card-actions">'
                '<div style="font-size:0.82rem; opacity:0.7; padding-top:0.4rem;">'
                'No action is currently needed.</div></div>',
                unsafe_allow_html=True,
            )

        # AI Understanding subsection (collapsible)
        render_column_ai_understanding(
            column,
            st.session_state.get("ai_understanding_result"),
        )

        # Separate expanders: 1) Column chart distribution, 2) Target analysis
        with st.expander("📊 Column chart distribution", expanded=False):
            render_column_distribution_chart(
                column,
                series,
                is_identifier=bool(id_finding),
            )

        with st.expander("🎯 Target analysis", expanded=False):
            render_target_relationship_analysis(
                column,
                series,
                df,
                target,
                findings,
                relevance.get(column),
                is_identifier=bool(id_finding),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Shared card sub-renderers (used by column_detail.py too via import)
# ─────────────────────────────────────────────────────────────────────────────


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


def _issue_title(finding: Finding) -> str:
    if finding.issue_type == "manual_review":
        return "Needs manual review"
    return finding.issue_type.replace("_", " ").title()


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
        title = _issue_title(finding)
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
