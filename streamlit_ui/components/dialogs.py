"""Training-result and feature-relevance dialogs with advanced metrics."""

from typing import Any, Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from data_doctor.report import compare_runs
from streamlit_ui.state import apply_working_action


# ─────────────────────────────────────────────────────────────────────────────
# Training result dialog
# ─────────────────────────────────────────────────────────────────────────────


@st.dialog("Training result", width="large")
def render_training_results() -> None:
    history = st.session_state.run_history
    if not history:
        return
    latest = history[-1]
    result = st.session_state.latest_training_result
    metrics = (result.get("metrics") if result else None) or latest.get("metrics") or latest
    pos_class = metrics.get("positive_class") or metrics.get("roc_auc_positive_class")
    is_binary = pos_class is not None

    st.subheader("Latest run results")
    model_type = (result or {}).get("model_type", "logistic_regression")
    model_label = (
        "Hist Gradient Boosting"
        if model_type == "hist_gradient_boosting"
        else "Logistic Regression"
    )
    st.caption(f"Model: **{model_label}**")

    # ── Threshold control (binary only) ──────────────────────────────────────
    threshold = 0.5
    if is_binary:
        threshold = st.slider(
            "Classification threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            help=(
                "Adjust the decision boundary. "
                "PR-AUC (Average Precision) is threshold-independent — "
                "it is calculated from the ranking of predicted scores."
            ),
        )

    # Recompute predictions at chosen threshold (binary only)
    result = st.session_state.latest_training_result
    if result and is_binary and result.get("test_probabilities"):
        y_test_raw = result["y_test"]
        prob_labels = result["test_prob_labels"]
        probs_array = np.array(result["test_probabilities"])
        pos_class = metrics["positive_class"]
        pos_idx = prob_labels.index(pos_class) if pos_class in prob_labels else 1
        pos_probs = probs_array[:, pos_idx]
        y_test_series = pd.Series(y_test_raw).astype(str)
        binary_truth = (y_test_series == pos_class).astype(int)
        threshold_preds = (pos_probs >= threshold).astype(int).tolist()
        threshold_pred_labels = [
            pos_class if p == 1 else [c for c in prob_labels if c != pos_class][0]
            for p in threshold_preds
        ]
        threshold_accuracy = float(accuracy_score(y_test_series, threshold_pred_labels))
        threshold_precision = float(
            precision_score(binary_truth, threshold_preds, zero_division=0)
        )
        threshold_recall = float(
            recall_score(binary_truth, threshold_preds, zero_division=0)
        )
        threshold_f1 = float(f1_score(binary_truth, threshold_preds, zero_division=0))
        threshold_cm = confusion_matrix(
            y_test_series,
            threshold_pred_labels,
            labels=metrics.get("labels", prob_labels),
        ).tolist()
    else:
        threshold_accuracy = metrics.get("accuracy", 0.0)
        threshold_precision = None
        threshold_recall = None
        threshold_f1 = None
        threshold_cm = metrics.get("confusion_matrix")
        threshold_pred_labels = None

    # ── Core metric tiles ─────────────────────────────────────────────────────
    n_cols = 4 if is_binary else 2
    metric_cols = st.columns(n_cols)
    metric_cols[0].metric(
        "Accuracy",
        f"{threshold_accuracy:.3f}",
        help="Fraction of correctly classified test examples.",
    )
    metric_cols[1].metric(
        "Macro F1",
        f"{metrics.get('macro_f1', 0):.3f}",
        help=(
            "Average F1 across classes, treating each class equally. "
            "More informative than accuracy when classes are imbalanced."
        ),
    )
    if is_binary:
        metric_cols[2].metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc") is not None else "—",
            help="Area under the ROC curve. 1.0 = perfect, 0.5 = random.",
        )
        metric_cols[3].metric(
            "Avg Precision (PR-AUC)",
            f"{metrics['average_precision']:.3f}"
            if metrics.get("average_precision") is not None
            else "—",
            help=(
                "Average Precision summarises the Precision-Recall curve. "
                "Particularly useful when the positive class is rare."
            ),
        )
        st.caption(
            f"ROC-AUC positive class: {pos_class} "
            f"(support: {sum(1 for v in (result or {}).get('y_test', []) if str(v) == pos_class):,} "
            f"of {len((result or {}).get('y_test', [])):,} test rows)"
        )

    for message in latest.get("warnings", []):
        st.warning(message)
    if metrics.get("macro_f1", 1) + 0.10 < metrics.get("accuracy", 0):
        st.warning(
            "Accuracy is noticeably higher than macro F1. "
            "The model may be weaker on a smaller class."
        )

    # ── Threshold-dependent binary metrics ───────────────────────────────────
    if is_binary and threshold_precision is not None:
        st.markdown("**Threshold-dependent metrics** (at current threshold)")
        t_cols = st.columns(3)
        t_cols[0].metric(
            "Precision",
            f"{threshold_precision:.3f}",
            help="Of all predicted positives, what fraction are truly positive?",
        )
        t_cols[1].metric(
            "Recall",
            f"{threshold_recall:.3f}",
            help="Of all actual positives, what fraction did the model find?",
        )
        t_cols[2].metric(
            "F1",
            f"{threshold_f1:.3f}",
            help="Harmonic mean of Precision and Recall.",
        )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_labels = metrics.get("labels", [])
    if threshold_cm and cm_labels:
        cm_df = pd.DataFrame(
            threshold_cm,
            index=[f"Actual {lbl}" for lbl in cm_labels],
            columns=[f"Predicted {lbl}" for lbl in cm_labels],
        )
        st.markdown("**Confusion matrix**")
        _, matrix_col, _ = st.columns([1, 2, 1])
        with matrix_col:
            st.dataframe(cm_df, width="stretch")

    # ── PR Curve (binary only) ────────────────────────────────────────────────
    if (
        is_binary
        and result
        and result.get("test_probabilities")
        and metrics.get("average_precision") is not None
    ):
        _render_pr_curve(result, metrics, pos_probs, binary_truth, threshold)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class = metrics.get("per_class_metrics", [])
    if per_class:
        st.markdown("**Per-class metrics**")
        pc_df = pd.DataFrame(per_class).rename(
            columns={
                "class": "Class",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "support": "Support",
            }
        )
        st.dataframe(
            pc_df.style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        if not is_binary:
            mac_col, wt_col = st.columns(2)
            mac_col.metric(
                "Macro Precision",
                f"{metrics.get('macro_precision', 0):.3f}",
            )
            wt_col.metric(
                "Weighted Precision",
                f"{metrics.get('weighted_precision', 0):.3f}",
            )

    # ── Run history ───────────────────────────────────────────────────────────
    st.markdown("**Run history**")
    visible = [
        "run_number",
        "model_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "roc_auc",
        "train_rows_before",
        "train_rows_after_cleaning",
        "train_rows_after_sampling",
        "open_issue_count",
    ]
    processed_history = []
    for item in history:
        row = dict(item)
        if "precision" not in row or row["precision"] is None:
            pc_list = row.get("per_class_metrics") or []
            pc_target = row.get("positive_class") or row.get("roc_auc_positive_class")
            if pc_list and pc_target:
                for pc in pc_list:
                    if str(pc.get("class")) == str(pc_target):
                        row["precision"] = pc.get("precision")
                        row["recall"] = pc.get("recall")
                        row["f1"] = pc.get("f1")
                        break
            if "precision" not in row or row["precision"] is None:
                row["precision"] = row.get("macro_precision")
                row["recall"] = row.get("macro_recall")
                row["f1"] = row.get("macro_f1")
        processed_history.append(row)

    history_df = pd.DataFrame(processed_history)
    available = [c for c in visible if c in history_df.columns]
    format_dict = {
        col: "{:.4f}"
        for col in ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
        if col in available
    }
    st.dataframe(
        history_df[available].style.format(format_dict, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    if len(history) >= 2:
        _render_run_comparison(history)
    _render_slice_diagnostics(latest)


def _render_pr_curve(
    result: dict[str, Any],
    metrics: dict[str, Any],
    pos_probs: np.ndarray,
    binary_truth: pd.Series,
    threshold: float,
) -> None:
    prec_arr, rec_arr, thresholds_arr = precision_recall_curve(
        binary_truth, pos_probs
    )
    prevalence = float(binary_truth.mean())
    ap = metrics.get("average_precision", 0.0)

    pr_df = pd.DataFrame({"Recall": rec_arr, "Precision": prec_arr})
    baseline_df = pd.DataFrame(
        {"Recall": [0.0, 1.0], "Precision": [prevalence, prevalence]}
    )

    pr_line = (
        alt.Chart(pr_df)
        .mark_line(color="#3478c7", strokeWidth=2)
        .encode(
            x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("Recall:Q", format=".3f"),
                alt.Tooltip("Precision:Q", format=".3f"),
            ],
        )
    )
    baseline_line = (
        alt.Chart(baseline_df)
        .mark_line(
            color="#9ca3af",
            strokeDash=[4, 4],
            strokeWidth=1.5,
        )
        .encode(x="Recall:Q", y="Precision:Q")
    )

    # Threshold point
    # Find the threshold index closest to selected threshold
    if len(thresholds_arr) > 0:
        closest_idx = int(np.argmin(np.abs(thresholds_arr - threshold)))
        thresh_pt_df = pd.DataFrame(
            {
                "Recall": [float(rec_arr[closest_idx])],
                "Precision": [float(prec_arr[closest_idx])],
            }
        )
        threshold_point = (
            alt.Chart(thresh_pt_df)
            .mark_point(color="#e5484d", size=80, filled=True)
            .encode(
                x="Recall:Q",
                y="Precision:Q",
                tooltip=[
                    alt.Tooltip("Recall:Q", title="Recall at threshold", format=".3f"),
                    alt.Tooltip("Precision:Q", title="Precision at threshold", format=".3f"),
                ],
            )
        )
        pr_chart = (pr_line + baseline_line + threshold_point).properties(
            height=260,
            title=alt.TitleParams(
                f"Precision-Recall Curve  (AP = {ap:.3f})",
                subtitle=f"Grey dashed = positive-class prevalence ({prevalence:.1%})  •  Red dot = current threshold ({threshold:.2f})",
                subtitleFontSize=11,
                subtitleColor="#6b7280",
            ),
        )
    else:
        pr_chart = (pr_line + baseline_line).properties(
            height=260,
            title=f"Precision-Recall Curve  (AP = {ap:.3f})",
        )

    st.markdown("**Precision-Recall curve**")
    st.altair_chart(pr_chart, width="stretch")


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
                wobble = comparison["metric_wobble"].get(key)
                metric_rows.append(
                    {
                        "Metric": label,
                        "Before": values["before"],
                        "After": values["after"],
                        "Change": values["change"],
                        "95% bootstrap range": (
                            f"{wobble['lower']:+.3f} to "
                            f"{wobble['upper']:+.3f}"
                            if wobble
                            else "Not available"
                        ),
                        "Wobble check": (
                            wobble["assessment"].replace("_", " ")
                            if wobble
                            else "Not available"
                        ),
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
        if comparison.get("roc_auc_note"):
            st.warning(comparison["roc_auc_note"])
        elif comparison.get("roc_auc") is not None:
            positive = history[-1].get("roc_auc_positive_class")
            if positive and positive == history[-2].get(
                "roc_auc_positive_class"
            ):
                st.caption(
                    f"Both ROC-AUC scores use {positive} as the positive class."
                )
        if comparison.get("held_out_rows_match") is False:
            st.caption(
                "The bootstrap range is hidden because the two runs did "
                "not score the same held-out rows."
            )
        if comparison["caption"]:
            st.caption(comparison["caption"])
        if comparison["wobble_caption"]:
            st.info(comparison["wobble_caption"])


def _render_slice_diagnostics(latest: dict[str, Any]) -> None:
    slices = latest.get("slice_diagnostics", [])
    if not slices:
        return
    st.markdown("**Slice diagnostics**")
    st.caption(
        "Test metrics for categorical groups with at least 10 rows and "
        "at least 5% of the test set. Macro F1 is shown only when every "
        "target class has at least 5 rows in the group. Smaller groups "
        "and thin classes are omitted. These are signals to inspect, "
        "not proof of unfairness."
    )
    table = pd.DataFrame(slices).rename(
        columns={
            "column": "Column",
            "value": "Value",
            "support": "Rows",
            "class_count": "Classes",
            "smallest_class_count": "Smallest class",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "note": "Note",
        }
    )
    if "Note" in table.columns and table["Note"].eq("").all():
        table = table.drop(columns=["Note"])
    st.dataframe(
        table.sort_values(
            ["Macro F1", "Accuracy"],
            na_position="last",
        ),
        hide_index=True,
        width="stretch",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature relevance dialog
# ─────────────────────────────────────────────────────────────────────────────


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
                    width="stretch",
                )


@st.dialog("Feature relevance", width="large")
def render_feature_relevance() -> None:
    result = st.session_state.latest_training_result
    if result is None:
        st.info("Train the model first.")
        return
    relevance = result["feature_relevance"]
    model_type = result.get("model_type", "logistic_regression")
    relevance_table = pd.DataFrame(
        {
            "Column": list(relevance),
            "Model influence": list(relevance.values()),
        }
    )
    relevance_table["Influence %"] = (
        relevance_table["Model influence"] * 100
    ).round(2)
    if model_type == "hist_gradient_boosting":
        st.caption(
            "For Hist Gradient Boosting, each percentage represents that "
            "column's share of permutation feature importance (mean decrease "
            "in test accuracy when that feature's values are shuffled). "
            "This measures non-linear predictive influence across trees. "
            "This is model influence, not proof that a feature causes the target."
        )
    else:
        st.caption(
            "Each percentage is that column's share of the model's absolute "
            "Logistic Regression coefficient weight. Numeric and date columns "
            "are scaled first, so a coefficient describes a one-standard-deviation "
            "change. A categorical percentage adds up that column's encoded "
            "categories, and a column with many categories can therefore look "
            "larger. Correlated columns can split one pattern across several "
            "percentages. This is model influence, not proof that a feature "
            "causes the target."
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
    st.altair_chart(chart, width="stretch")
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
    current_columns = [column for column in available if column not in selected]
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
        width="stretch",
    ):
        for column in selected:
            apply_working_action("drop_column", column=column)
        st.session_state["_last_action_success"] = (
            f"Dropped column(s): **{', '.join(selected)}**."
        )
        st.rerun()
    if cancel_column.button(
        "Cancel",
        type="tertiary",
        width="stretch",
    ):
        st.session_state.feature_drop_selection = []
        st.rerun()
