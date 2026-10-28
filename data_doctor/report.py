"""Templates for saved runs and factual between-run comparisons."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_run_record(
    training_result: dict[str, Any],
    df: pd.DataFrame,
    findings: list[Any],
    action_log_position: int,
) -> dict[str, Any]:
    metrics = training_result["metrics"]
    return {
        "run_number": 0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
        "train_rows_before": training_result["train_rows_before"],
        "train_rows_after": training_result["train_rows_after"],
        "test_rows": training_result["test_rows"],
        "data_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "open_issue_count": int(
            sum(bool(getattr(finding, "found", False)) for finding in findings)
        ),
        "action_log_position": int(action_log_position),
        "class_weight": training_result["class_weight"],
        "sampling": training_result["sampling"],
    }


def _metric_change(before: float, after: float) -> dict[str, float]:
    return {
        "before": float(before),
        "after": float(after),
        "change": float(after - before),
    }


def compare_runs(
    previous: dict[str, Any],
    current: dict[str, Any],
    action_log: list[dict[str, Any]],
) -> dict[str, Any]:
    actions = action_log[
        previous["action_log_position"] : current["action_log_position"]
    ]
    previous_columns = set(previous["columns"])
    current_columns = set(current["columns"])
    facts = {
        "actions": actions,
        "columns_added": sorted(current_columns - previous_columns),
        "columns_removed": sorted(previous_columns - current_columns),
        "train_rows": {
            "before": int(previous["train_rows_after"]),
            "after": int(current["train_rows_after"]),
            "change": int(
                current["train_rows_after"]
                - previous["train_rows_after"]
            ),
        },
        "accuracy": _metric_change(
            previous["accuracy"], current["accuracy"]
        ),
        "macro_f1": _metric_change(
            previous["macro_f1"], current["macro_f1"]
        ),
        "roc_auc": None,
    }
    if (
        previous.get("roc_auc") is not None
        and current.get("roc_auc") is not None
    ):
        facts["roc_auc"] = _metric_change(
            previous["roc_auc"], current["roc_auc"]
        )
    facts["caption"] = interpretation_caption(
        previous, current, actions, facts
    )
    return facts


def interpretation_caption(
    previous: dict[str, Any],
    current: dict[str, Any],
    actions: list[dict[str, Any]],
    facts: dict[str, Any],
) -> str | None:
    action_names = {action["action"] for action in actions}
    accuracy_change = facts["accuracy"]["change"]
    f1_change = facts["macro_f1"]["change"]

    dropped_columns = [
        action
        for action in actions
        if action["action"] == "drop_column"
    ]
    if dropped_columns and accuracy_change <= -0.05:
        return (
            "The earlier score may have been too optimistic because "
            "accuracy fell after a column was removed."
        )
    if dropped_columns and abs(accuracy_change) < 0.01:
        return "The removed column was not materially helping the score."
    if (
        current.get("class_weight") == "balanced"
        and f1_change > 0.01
        and abs(accuracy_change) < 0.01
    ):
        return (
            "Balanced class weights improved macro F1 while accuracy "
            "stayed about the same."
        )
    if facts["train_rows"]["change"] < -0.10 * previous["train_rows_after"]:
        if accuracy_change > 0.03:
            return (
                "The score rose after many rows were removed; those rows "
                "may have contained harder examples."
            )
    if {
        "impute_mean",
        "impute_median",
        "impute_mode",
        "fill_constant",
    } & action_names and abs(accuracy_change) < 0.01:
        return "Missingness was probably not the main performance problem."
    return None

