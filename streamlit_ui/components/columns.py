"""Per-column analysis cards with inline visualizations and pagination."""

from typing import Any

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding
from streamlit_ui.actions import (
    coerce_label_value,
    missing_like_label_values,
)
from streamlit_ui.components.common import highest_severity, render_severity
from streamlit_ui.config import SEVERITY_ORDER
from streamlit_ui.state import apply_working_action
from streamlit_ui.visualizations.categorical_charts import (
    compact_frequency_chart,
    detail_frequency_chart,
)
from streamlit_ui.visualizations.numeric_charts import (
    compact_histogram,
    detail_histogram,
)
from streamlit_ui.visualizations.target_analysis import (
    categorical_target_distribution,
    numeric_target_distribution,
)


def render_column_grid(
    df: pd.DataFrame,
    target: str,
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    title_column, toggle_column = st.columns([4, 1])
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
        "and available actions together. Click 'View details' for charts."
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
    page_size = st.selectbox(
        "Columns per page",
        [12, 24, 48],
        index=0,
        key="column_page_size",
    )
    page_count = max(1, (len(column_names) + page_size - 1) // page_size)
    page = int(
        st.number_input(
            "Column page",
            min_value=1,
            max_value=page_count,
            value=1,
            key="column_page",
        )
    )
    visible_columns = column_names[
        (page - 1) * page_size : page * page_size
    ]
    st.caption(
        f"Showing {len(visible_columns)} of {len(column_names)} columns "
        f"(page {page} of {page_count})."
    )
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
        st.markdown(f"#### {column}{relevance_suffix}")
        st.caption("Target column" if column == target else str(series.dtype))
        _render_column_information(series, bool(id_finding))

        # Compact chart preview (not for ID-like or target columns)
        if not id_finding and column != target:
            _render_compact_chart(series)

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
            _render_column_actions(column, series, applicable)
        elif column == target:
            st.caption("Target labels can be reviewed but not dropped.")
        else:
            st.caption("No action is currently needed.")

        # View details expander with full charts (lazy — only rendered when opened)
        if column != target and not id_finding:
            with st.expander("View details"):
                _render_detail_charts(column, series, df, target)


def _render_compact_chart(series: pd.Series) -> None:
    """Render a small inline chart preview inside the column card."""
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            chart = compact_histogram(series)
        else:
            chart = compact_frequency_chart(series)
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        # Never let a chart error break the card
        pass


def _render_detail_charts(
    column: str,
    series: pd.Series,
    df: pd.DataFrame,
    target: str,
) -> None:
    """Render full-size distribution chart + target-distribution chart."""
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            st.altair_chart(
                detail_histogram(series, column_name=column),
                use_container_width=True,
            )
        else:
            st.altair_chart(
                detail_frequency_chart(series, column_name=column),
                use_container_width=True,
            )
    except Exception:
        pass

    if target not in df.columns:
        st.caption("Select a target column to see feature-vs-target charts.")
        return

    target_series = df[target]
    st.markdown("**Target distribution**")
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
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Not enough data to show a target distribution chart.")
    except Exception:
        st.caption("Target distribution chart unavailable for this column.")


def _render_column_information(
    series: pd.Series,
    is_identifier: bool,
) -> None:
    st.markdown(
        '<div class="dd-section-label">Information</div>',
        unsafe_allow_html=True,
    )
    info = [
        f"Missing: {series.isna().sum():,} ({series.isna().mean():.1%})",
        f"Unique: {series.nunique(dropna=True):,}",
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
    st.write("  \n".join(info))


def _render_column_problems(issues: list[Finding]) -> None:
    st.markdown(
        '<div class="dd-section-label">Problems</div>',
        unsafe_allow_html=True,
    )
    if not issues:
        st.success("No issue found")
        return
    for finding in issues:
        st.write(f"• {finding.issue_type.replace('_', ' ').title()}")
        st.caption(finding.reason)


def _render_column_actions(
    column: str,
    series: pd.Series,
    applicable: dict[str, dict[str, Any]],
) -> None:
    selected = st.selectbox(
        "Choose an action",
        list(applicable),
        format_func=lambda key: applicable[key]["label"],
        key=f"column_action_{column}",
        label_visibility="collapsed",
    )
    option = applicable[selected]
    value: Any = None
    old_label: Any = None
    if option["action"] == "fill_constant":
        value = st.text_input("Constant value", key=f"constant_{column}")
    elif option["action"] == "recode_label":
        old_label = st.selectbox(
            "Label to replace",
            missing_like_label_values(series),
            key=f"recode_old_{column}",
        )
        value = st.text_input(
            "New label value",
            key=f"recode_new_{column}",
        )
    disabled = (
        option["action"] in {"fill_constant", "recode_label"} and value == ""
    )
    action_value = value
    if option["action"] == "recode_label" and not disabled:
        try:
            action_value = {
                "old": old_label,
                "new": coerce_label_value(value, series),
            }
        except (ValueError, TypeError):
            action_value = value
    if st.button(
        "Apply action",
        key=f"column_apply_{column}",
        disabled=disabled,
        use_container_width=True,
    ):
        try:
            if option["action"] == "recode_label":
                action_value = {
                    "old": old_label,
                    "new": coerce_label_value(value, series),
                }
            apply_working_action(
                option["action"],
                column=column,
                value=action_value,
            )
            st.rerun()
        except (ValueError, TypeError) as error:
            st.error(str(error))
