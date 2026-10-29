"""Training-result and feature-relevance dialogs."""

from typing import Any, Callable

import altair as alt
import pandas as pd
import streamlit as st

from data_doctor.report import compare_runs
from streamlit_ui.state import apply_working_action


@st.dialog("Training result", width="large")
def render_training_results() -> None:
    history = st.session_state.run_history
    if not history:
        return
    latest = history[-1]
    st.subheader("Latest run results")
    metric_columns = st.columns(3 if latest["roc_auc"] is not None else 2)
    metric_columns[0].metric("Accuracy", f"{latest['accuracy']:.3f}")
    metric_columns[1].metric("Macro F1", f"{latest['macro_f1']:.3f}")
    if latest["roc_auc"] is not None:
        metric_columns[2].metric("ROC-AUC", f"{latest['roc_auc']:.3f}")
        st.caption(
            "ROC-AUC positive class: "
            f"{latest.get('roc_auc_positive_class', 'not recorded')}"
        )
    for message in latest.get("warnings", []):
        st.warning(message)
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
        "train_rows_before",
        "train_rows_after_cleaning",
        "train_rows_after_sampling",
        "open_issue_count",
    ]
    st.dataframe(
        pd.DataFrame(history)[visible],
        hide_index=True,
        width="stretch",
    )
    if len(history) >= 2:
        _render_run_comparison(history)
    _render_slice_diagnostics(latest)


def _render_run_comparison(history: list[dict[str, Any]]) -> None:
    comparison = compare_runs(
        history[-2],
        history[-1],
        st.session_state.action_log,
    )
    with st.expander("What changed since the previous train"):
        actions = comparison["actions"]
        st.markdown("**Actions applied**")
        st.write(
            ", ".join(
                action["action"].replace("_", " ")
                + (
                    f" ({action['column']})"
                    if action.get("column")
                    else ""
                )
                for action in actions
            )
            if actions
            else "No working-data actions."
        )
        if comparison["columns_removed"]:
            st.write(
                "Columns removed: "
                + ", ".join(comparison["columns_removed"])
            )
        if comparison["columns_added"]:
            st.write(
                "Columns added: " + ", ".join(comparison["columns_added"])
            )
        metric_rows = []
        for key, label in (
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro F1"),
            ("roc_auc", "ROC-AUC"),
        ):
            values = comparison[key]
            if values is not None:
                metric_rows.append(
                    {
                        "Metric": label,
                        "Before": values["before"],
                        "After": values["after"],
                        "Change": values["change"],
                    }
                )
        st.dataframe(
            pd.DataFrame(metric_rows),
            hide_index=True,
            width="stretch",
        )
        rows = comparison["train_rows"]
        st.caption(
            "Training rows after cleaning: "
            f"{rows['before']:,} → {rows['after']:,} "
            f"({rows['change']:+,})"
        )
        options = comparison["training_options"]
        if any(
            values["before"] != values["after"]
            for values in options.values()
        ):
            st.caption(
                "Training options changed: "
                + "; ".join(
                    f"{name.replace('_', ' ')} "
                    f"{values['before'] or 'None'} → "
                    f"{values['after'] or 'None'}"
                    for name, values in options.items()
                    if values["before"] != values["after"]
                )
            )
        if comparison["caption"]:
            st.caption(comparison["caption"])


def _render_slice_diagnostics(latest: dict[str, Any]) -> None:
    slices = latest.get("slice_diagnostics", [])
    if not slices:
        return
    st.markdown("**Slice diagnostics**")
    st.caption(
        "Test metrics for sufficiently large categorical groups. "
        "Small groups are omitted; differences are signals to inspect, "
        "not proof of unfairness."
    )
    st.dataframe(
        pd.DataFrame(slices).sort_values("macro_f1"),
        hide_index=True,
        width="stretch",
    )


def _move_feature_to_drop(column: str) -> None:
    selected = st.session_state.setdefault("feature_drop_selection", [])
    if column not in selected:
        selected.append(column)


def _restore_feature_column(column: str) -> None:
    selected = st.session_state.setdefault("feature_drop_selection", [])
    if column in selected:
        selected.remove(column)


def _feature_button_grid(
    columns: list[str],
    *,
    key_prefix: str,
    button_type: str,
    callback: Callable[[str], None],
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
def render_feature_relevance() -> None:
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
    chart_height = min(700, max(260, len(relevance_table) * 24))
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
                axis=alt.Axis(labelOverlap=False, labelLimit=180),
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
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        relevance_table[["Column", "Influence %"]],
        hide_index=True,
        width="stretch",
    )
    _render_feature_selection(relevance)


def _render_feature_selection(relevance: dict[str, float]) -> None:
    available = [
        column
        for column in relevance
        if column in st.session_state.working_df.columns
        and column != st.session_state.target
    ]
    selected = [
        column
        for column in st.session_state.get("feature_drop_selection", [])
        if column in available
    ]
    st.session_state.feature_drop_selection = selected
    current_columns = [
        column for column in available if column not in selected
    ]
    current_panel, drop_panel = st.columns(2)
    with current_panel:
        st.markdown("#### Current columns")
        st.caption("Click a green column to move it to Drop columns.")
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
            "Red columns are selected for removal. Click one to move it back."
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
            apply_working_action("drop_column", column=column)
        st.rerun()
    if cancel_column.button(
        "Cancel",
        type="tertiary",
        use_container_width=True,
    ):
        st.session_state.feature_drop_selection = []
        st.rerun()
