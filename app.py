"""Dashboard-style Streamlit frontend for DataDoctor."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from data_doctor.detectors import (
    Finding,
    dataset_vitals,
    diagnose,
    id_like_columns,
    prepare_data,
)
from data_doctor.report import build_run_record, compare_runs
from data_doctor.treatments import apply_action
from notebooks.model_training import train_model

ROOT = Path(__file__).resolve().parent
DEMO_FILES = {
    "Messy Telco churn": ROOT / "data" / "Messy" / "telco_messy.csv",
    "Messy Adult income": ROOT / "data" / "Messy" / "adult_messy.csv",
    "Messy credit-card fraud": (
        ROOT / "data" / "Messy" / "creditcard_messy.csv"
    ),
}
DEMO_TARGETS = {
    "Messy Telco churn": "Churn",
    "Messy Adult income": "income",
    "Messy credit-card fraud": "Class",
}
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

st.set_page_config(
    page_title="DataDoctor",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.7rem;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.85rem;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.82rem;
        white-space: normal;
    }
    .control-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    [data-testid="stDialog"] button[kind="secondary"] {
        background: #123d2a;
        color: #8ff0b8;
        border-color: #287a50;
    }
    [data-testid="stDialog"] button[kind="secondary"]:hover {
        background: #185238;
        color: #b9ffd3;
        border-color: #36a469;
    }
    .st-key-training_result_button button:not(:disabled) {
        background: #1f7a4d;
        color: white;
        border-color: #2d9b65;
    }
    .st-key-training_result_button button:not(:disabled):hover {
        background: #286f4d;
        color: white;
        border-color: #3eaf76;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _read_demo(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _source_key(name: str, content: bytes | None = None) -> str:
    digest = hashlib.sha256(content).hexdigest() if content else name
    return f"{name}:{digest}"


def _reset_for_source(
    df: pd.DataFrame,
    key: str,
    suggested_target: str | None = None,
) -> None:
    st.session_state.source_key = key
    st.session_state.original_df = df.copy(deep=True)
    st.session_state.working_df = df.copy(deep=True)
    st.session_state.target = suggested_target
    st.session_state.action_log = []
    st.session_state.run_history = []
    st.session_state.findings = None
    st.session_state.latest_training_result = None
    st.session_state.last_feature_relevance = {}
    st.session_state.show_relevance_scores = False
    st.session_state.feature_drop_selection = []


def _reset_for_target(target: str) -> None:
    st.session_state.working_df = (
        st.session_state.original_df.copy(deep=True)
    )
    st.session_state.target = target
    st.session_state.action_log = []
    st.session_state.run_history = []
    st.session_state.findings = None
    st.session_state.latest_training_result = None
    st.session_state.last_feature_relevance = {}
    st.session_state.show_relevance_scores = False
    st.session_state.feature_drop_selection = []


def _refresh_findings() -> list[Finding]:
    findings = diagnose(
        st.session_state.working_df,
        st.session_state.target,
    )
    st.session_state.findings = findings
    return findings


def _apply(
    action: str,
    column: str | None = None,
    value: Any = None,
) -> None:
    updated, log = apply_action(
        st.session_state.working_df,
        st.session_state.target,
        action,
        column=column,
        value=value,
    )
    st.session_state.working_df = updated
    st.session_state.action_log.append(log)
    st.session_state.findings = None
    st.session_state.latest_training_result = None
    st.session_state.feature_drop_selection = []


def _severity_label(severity: str) -> str:
    return {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "none": "No issue found",
    }.get(severity, severity.title())


def _render_severity(severity: str) -> None:
    label = _severity_label(severity)
    if severity == "high":
        st.error(label)
    elif severity == "medium":
        st.warning(label)
    elif severity == "low":
        st.info(label)
    else:
        st.success(label)


def _highest_severity(findings: list[Finding]) -> str:
    if not findings:
        return "none"
    return max(
        (finding.severity for finding in findings),
        key=lambda value: SEVERITY_ORDER.get(value, 0),
    )


def _action_options(
    findings: list[Finding],
    df: pd.DataFrame,
    target: str,
) -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}

    def add(label: str, action: str, column: str | None = None) -> None:
        key = f"{action}|{column or 'dataset'}"
        options[key] = {
            "label": label,
            "action": action,
            "column": column,
        }

    for finding in findings:
        if not finding.found:
            continue
        column = finding.scope if finding.scope != "dataset" else None
        if finding.issue_type == "missing_values" and column:
            if column == target:
                add(
                    f"Drop rows with missing {target}",
                    "drop_rows_missing",
                    column,
                )
            elif finding.evidence["percent"] >= 80:
                add(f"Drop {column}", "drop_column", column)
            else:
                if pd.api.types.is_numeric_dtype(df[column]):
                    add(
                        f"Fill {column} with median",
                        "impute_median",
                        column,
                    )
                    add(
                        f"Fill {column} with mean",
                        "impute_mean",
                        column,
                    )
                else:
                    add(
                        f"Fill {column} with mode",
                        "impute_mode",
                        column,
                    )
                add(
                    f"Fill {column} with a constant",
                    "fill_constant",
                    column,
                )
                add(
                    f"Drop rows missing {column}",
                    "drop_rows_missing",
                    column,
                )
                add(f"Drop {column}", "drop_column", column)
        elif finding.issue_type == "exact_duplicates":
            add("Drop exact duplicate rows", "drop_exact_duplicates")
        elif finding.issue_type == "conflicting_labels":
            add(
                "Drop conflicting-label rows",
                "drop_conflicting_rows",
            )
        elif finding.issue_type == "outliers" and column:
            add(f"Cap outliers in {column}", "cap_outliers", column)
            add(
                f"Remove outlier rows for {column}",
                "remove_outliers",
                column,
            )
        elif finding.issue_type in {"id_like", "target_copy"} and column:
            if column != target:
                add(f"Drop {column}", "drop_column", column)
        elif finding.issue_type == "weird_labels":
            add("Drop missing-like target rows", "drop_weird_label_rows")

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        if column == target:
            continue
        shares = df[column].value_counts(normalize=True, dropna=True)
        if len(shares[shares < 0.01]):
            add(
                f"Group rare {column} values as Other",
                "group_rare_categories",
                column,
            )
    return options


def _show_vitals(df: pd.DataFrame) -> None:
    vitals = dataset_vitals(prepare_data(df))
    columns = st.columns(4)
    columns[0].metric("Rows", f"{vitals['rows']:,}")
    columns[1].metric("Columns", f"{vitals['columns']:,}")
    columns[2].metric(
        "Missing cells",
        f"{vitals['missing_cells']:,}",
        help=f"{vitals['missing_percent']:.2f}% of all cells",
    )
    columns[3].metric(
        "Duplicate extras",
        f"{vitals['exact_duplicate_extras']:,}",
    )


def _render_dataset_issues(
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    dataset_findings = [
        finding
        for finding in findings
        if finding.scope == "dataset"
    ]
    st.subheader("Dataset-wide issues")
    columns = st.columns(max(1, len(dataset_findings)))
    for container, finding in zip(columns, dataset_findings):
        with container:
            # The cards begin below the vitals/preview, so this height makes
            # their bottom edge align with the top-row Controls card.
            with st.container(border=True, height=588):
                st.markdown(
                    f"**{finding.issue_type.replace('_', ' ').title()}**"
                )
                st.markdown("**Severity**")
                _render_severity(finding.severity)
                st.caption(finding.reason)
                if finding.found:
                    evidence_text = " · ".join(
                        f"{key.replace('_', ' ').title()}: {value}"
                        for key, value in finding.evidence.items()
                    )
                    st.caption(evidence_text)
                expected_action = {
                    "exact_duplicates": "drop_exact_duplicates",
                    "conflicting_labels": "drop_conflicting_rows",
                }.get(finding.issue_type)
                applicable = {
                    key: option
                    for key, option in options.items()
                    if option["column"] is None
                    and option["action"] == expected_action
                }
                if applicable and finding.found:
                    selected = st.selectbox(
                        "Action",
                        list(applicable),
                        format_func=lambda key, mapping=applicable: (
                            mapping[key]["label"]
                        ),
                        key=f"dataset_action_{finding.issue_type}",
                    )
                    if st.button(
                        "Apply",
                        key=f"dataset_apply_{finding.issue_type}",
                    ):
                        option = applicable[selected]
                        _apply(option["action"])
                        st.rerun()


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
    severity = _highest_severity(column_findings)
    applicable = {
        key: option
        for key, option in options.items()
        if option["column"] == column
    }

    with st.container(border=True, height=520):
        relevance = (
            st.session_state.get("last_feature_relevance", {})
            if st.session_state.get("show_relevance_scores", False)
            else {}
        )
        relevance_suffix = (
            f" ({relevance[column]:.2%})"
            if column in relevance
            else ""
        )
        st.markdown(f"#### {column}{relevance_suffix}")
        st.caption("Target column" if column == target else str(series.dtype))

        st.markdown("**Information**")
        info = [
            f"Missing: {series.isna().sum():,} ({series.isna().mean():.1%})",
            f"Unique: {series.nunique(dropna=True):,}",
        ]
        if (
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

        st.markdown("**Problems**")
        if issues:
            for finding in issues:
                st.write(
                    f"• {finding.issue_type.replace('_', ' ').title()}"
                )
                st.caption(finding.reason)
        else:
            st.success("No issue found")

        st.markdown("**Severity**")
        _render_severity(severity)

        st.markdown("**Actions**")
        if applicable:
            selected = st.selectbox(
                "Choose an action",
                list(applicable),
                format_func=lambda key: applicable[key]["label"],
                key=f"column_action_{column}",
                label_visibility="collapsed",
            )
            option = applicable[selected]
            value: Any = None
            if option["action"] == "fill_constant":
                value = st.text_input(
                    "Constant value",
                    key=f"constant_{column}",
                )
            disabled = option["action"] == "fill_constant" and value == ""
            if st.button(
                "Apply action",
                key=f"column_apply_{column}",
                disabled=disabled,
                use_container_width=True,
            ):
                try:
                    _apply(
                        option["action"],
                        column=column,
                        value=value,
                    )
                    st.rerun()
                except (ValueError, TypeError) as error:
                    st.error(str(error))
        elif column == target:
            st.caption("Target labels can be reviewed but not dropped.")
        else:
            st.caption("No action is currently needed.")


def _render_column_grid(
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
        "and available actions together."
    )
    original_order = {
        column: index for index, column in enumerate(df.columns)
    }
    severity_by_column = {
        column: _highest_severity(
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
    for start in range(0, len(column_names), 4):
        row = st.columns(4)
        for container, column in zip(
            row, column_names[start : start + 4]
        ):
            with container:
                _render_column_card(
                    column,
                    df,
                    target,
                    findings,
                    options,
                )


@st.dialog("Training result", width="large")
def _render_training_results() -> None:
    history = st.session_state.run_history
    if not history:
        return
    latest = history[-1]
    st.subheader("Latest run results")
    metric_columns = st.columns(
        3 if latest["roc_auc"] is not None else 2
    )
    metric_columns[0].metric("Accuracy", f"{latest['accuracy']:.3f}")
    metric_columns[1].metric("Macro F1", f"{latest['macro_f1']:.3f}")
    if latest["roc_auc"] is not None:
        metric_columns[2].metric("ROC-AUC", f"{latest['roc_auc']:.3f}")
    if latest["macro_f1"] + 0.10 < latest["accuracy"]:
        st.warning(
            "Accuracy is noticeably higher than macro F1. "
            "The model may be weaker on a smaller class."
        )

    matrix = pd.DataFrame(
        latest["confusion_matrix"],
        index=[f"Actual {label}" for label in latest["labels"]],
        columns=[f"Predicted {label}" for label in latest["labels"]],
    )
    st.markdown("**Confusion matrix**")
    _, matrix_column, _ = st.columns([1, 2, 1])
    with matrix_column:
        st.dataframe(matrix, width="stretch")

    st.markdown("**Run history**")
    visible = [
        "run_number",
        "accuracy",
        "macro_f1",
        "roc_auc",
        "train_rows_after",
        "open_issue_count",
    ]
    st.dataframe(
        pd.DataFrame(history)[visible],
        hide_index=True,
        width="stretch",
    )

    if len(history) >= 2:
        comparison = compare_runs(
            history[-2],
            history[-1],
            st.session_state.action_log,
        )
        with st.expander("What changed since the previous train"):
            st.json(
                {
                    key: value
                    for key, value in comparison.items()
                    if key != "caption"
                }
            )
            if comparison["caption"]:
                st.caption(comparison["caption"])


def _move_feature_to_drop(column: str) -> None:
    selected = st.session_state.setdefault(
        "feature_drop_selection", []
    )
    if column not in selected:
        selected.append(column)


def _restore_feature_column(column: str) -> None:
    selected = st.session_state.setdefault(
        "feature_drop_selection", []
    )
    if column in selected:
        selected.remove(column)


def _feature_button_grid(
    columns: list[str],
    *,
    key_prefix: str,
    button_type: str,
    callback: Any,
    relevance: dict[str, float],
) -> None:
    if not columns:
        st.caption("No columns.")
        return
    for start in range(0, len(columns), 2):
        row = st.columns(2)
        for container, column in zip(row, columns[start : start + 2]):
            with container:
                st.button(
                    column,
                    key=f"{key_prefix}_{column}",
                    type=button_type,
                    help=f"Model influence: {relevance[column]:.2%}",
                    on_click=callback,
                    args=(column,),
                    use_container_width=True,
                )


@st.dialog("Feature relevance", width="large")
def _feature_relevance_dialog() -> None:
    result = st.session_state.latest_training_result
    if result is None:
        st.info("Train the model first.")
        return
    relevance = result["feature_relevance"]
    relevance_table = pd.DataFrame(
        {
            "Column": list(relevance),
            "Model influence": list(relevance.values()),
        }
    )
    relevance_table["Influence %"] = (
        relevance_table["Model influence"] * 100
    ).round(2)
    st.caption(
        "Influence is based on absolute Logistic Regression coefficients "
        "after preprocessing. It is not proof that a feature causes the "
        "target."
    )
    chart_height = min(
        700, max(260, len(relevance_table) * 24)
    )
    chart = (
        alt.Chart(relevance_table)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model influence:Q",
                title="Model influence",
                scale=alt.Scale(domain=[0, 1], clamp=True),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "Column:N",
                sort="-x",
                title=None,
                axis=alt.Axis(
                    labelOverlap=False,
                    labelLimit=180,
                ),
            ),
            tooltip=[
                alt.Tooltip("Column:N"),
                alt.Tooltip(
                    "Model influence:Q",
                    title="Influence",
                    format=".2%",
                ),
            ],
        )
        .properties(height=chart_height)
        .configure_view(strokeWidth=0)
    )
    # No interactive selection is attached, so pan and zoom stay disabled.
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        relevance_table[["Column", "Influence %"]],
        hide_index=True,
        width="stretch",
    )

    available = [
        column
        for column in relevance
        if column in st.session_state.working_df.columns
        and column != st.session_state.target
    ]
    selected = [
        column
        for column in st.session_state.get(
            "feature_drop_selection", []
        )
        if column in available
    ]
    st.session_state.feature_drop_selection = selected
    current_columns = [
        column for column in available if column not in selected
    ]
    current_panel, drop_panel = st.columns(2)
    with current_panel:
        st.markdown("#### Current columns")
        st.caption(
            "Click a green column to move it to Drop columns."
        )
        _feature_button_grid(
            current_columns,
            key_prefix="keep_feature",
            button_type="secondary",
            callback=_move_feature_to_drop,
            relevance=relevance,
        )
    with drop_panel:
        st.markdown("#### Drop columns")
        st.caption(
            "Click a dropped column to move it back."
        )
        _feature_button_grid(
            selected,
            key_prefix="drop_feature",
            button_type="primary",
            callback=_restore_feature_column,
            relevance=relevance,
        )
    save_column, cancel_column = st.columns(2)
    if save_column.button(
        "Save changes",
        type="primary",
        disabled=not selected,
        use_container_width=True,
    ):
        for column in selected:
            _apply("drop_column", column=column)
        st.rerun()
    if cancel_column.button(
        "Cancel",
        type="tertiary",
        use_container_width=True,
    ):
        st.session_state.feature_drop_selection = []
        st.rerun()


def _render_control_panel(
    target_options: list[str],
    findings: list[Finding],
) -> None:
    # Fixed together with the issue-card height so the two top-row sides
    # share a bottom edge even when a wider viewport wraps less content.
    with st.container(border=True, height=827):
        st.markdown(
            '<div class="control-title">Controls</div>',
            unsafe_allow_html=True,
        )
        current_target = st.session_state.target
        selected_target = st.selectbox(
            "Target column",
            target_options,
            index=(
                target_options.index(current_target)
                if current_target in target_options
                else 0
            ),
            help="ID-like columns are excluded.",
        )
        if selected_target != current_target:
            _reset_for_target(selected_target)
            st.rerun()

        st.divider()
        st.markdown("**Model training**")
        strategy = st.selectbox(
            "Class imbalance option",
            [
                "None",
                'class_weight="balanced"',
                "Random oversample",
                "Random undersample",
            ],
        )
        class_weight = (
            "balanced"
            if strategy == 'class_weight="balanced"'
            else None
        )
        sampling = {
            "Random oversample": "oversample",
            "Random undersample": "undersample",
        }.get(strategy)
        if st.button(
            "Train Logistic Regression",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Training…"):
                    result = train_model(
                        st.session_state.original_df,
                        selected_target,
                        class_weight=class_weight,
                        sampling=sampling,
                        actions=st.session_state.action_log,
                    )
                record = build_run_record(
                    result,
                    st.session_state.working_df,
                    findings,
                    len(st.session_state.action_log),
                )
                record["run_number"] = (
                    len(st.session_state.run_history) + 1
                )
                st.session_state.run_history.append(record)
                st.session_state.latest_training_result = result
                st.session_state.last_feature_relevance = dict(
                    result["feature_relevance"]
                )
                st.rerun()
            except ValueError as error:
                st.error(str(error))

        result_available = bool(st.session_state.run_history)
        if st.button(
            "Training result",
            key="training_result_button",
            disabled=not result_available,
            use_container_width=True,
        ):
            _render_training_results()

        trained = st.session_state.latest_training_result is not None
        if st.button(
            "Feature relevance",
            disabled=not trained,
            use_container_width=True,
        ):
            _feature_relevance_dialog()
        if not result_available:
            st.caption(
                "Train once to unlock the training result and feature "
                "relevance."
            )
        elif not trained:
            st.caption(
                "Train again to refresh feature relevance after your "
                "latest data change."
            )

        st.divider()
        st.markdown("**Working data**")
        st.download_button(
            "Download working CSV",
            st.session_state.working_df.to_csv(index=False).encode(
                "utf-8"
            ),
            file_name="datadoctor_working.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Reset to original", use_container_width=True):
            _reset_for_target(selected_target)
            st.rerun()
        if st.session_state.action_log:
            with st.expander(
                f"Action log ({len(st.session_state.action_log)})"
            ):
                st.dataframe(
                    pd.DataFrame(st.session_state.action_log),
                    hide_index=True,
                    width="stretch",
                )


def _load_source() -> tuple[pd.DataFrame | None, str | None, str | None]:
    source_mode = st.radio(
        "Data source",
        ["Demo dataset", "Upload CSV"],
        horizontal=True,
    )
    if source_mode == "Demo dataset":
        demo_name = st.selectbox("Demo", list(DEMO_FILES))
        path = DEMO_FILES[demo_name]
        return (
            _read_demo(str(path)),
            _source_key(str(path)),
            DEMO_TARGETS[demo_name],
        )

    upload = st.file_uploader("Upload a CSV", type=["csv"])
    if upload is None:
        return None, None, None
    content = upload.getvalue()
    try:
        return (
            pd.read_csv(upload),
            _source_key(upload.name, content),
            None,
        )
    except Exception as error:
        st.error(f"Could not read this CSV: {error}")
        return None, None, None


def main() -> None:
    st.title("DataDoctor")
    st.caption(
        "Inspect a classification CSV, choose changes, and compare "
        "Logistic Regression runs. Nothing is cleaned automatically."
    )

    loaded_df, source_key, suggested_target = _load_source()
    if loaded_df is None or source_key is None:
        st.info("Choose a demo or upload a CSV to begin.")
        return
    if st.session_state.get("source_key") != source_key:
        _reset_for_source(loaded_df, source_key, suggested_target)

    working = st.session_state.working_df
    prepared = prepare_data(working)
    blocked_targets = set(id_like_columns(prepared))
    target_options = [
        column for column in prepared.columns if column not in blocked_targets
    ]
    if not target_options:
        st.error("No target candidate remains after excluding ID columns.")
        return
    if st.session_state.target not in target_options:
        st.session_state.target = target_options[0]

    try:
        findings = (
            st.session_state.findings
            if st.session_state.findings is not None
            else _refresh_findings()
        )
    except ValueError as error:
        st.error(str(error))
        return

    options = _action_options(
        findings,
        working,
        st.session_state.target,
    )
    main_column, control_column = st.columns([4.1, 1.4], gap="large")
    with control_column:
        _render_control_panel(target_options, findings)
    with main_column:
        _show_vitals(working)
        with st.expander("Preview data"):
            st.dataframe(working.head(10), width="stretch")
        _render_dataset_issues(findings, options)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _render_column_grid(
        working,
        st.session_state.target,
        findings,
        options,
    )


if __name__ == "__main__":
    main()
