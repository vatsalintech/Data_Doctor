"""Templates for saved runs and factual between-run comparisons."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from data_doctor.detectors import dataset_vitals, prepare_data


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
        "roc_auc_positive_class": metrics.get(
            "roc_auc_positive_class"
        ),
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
        "train_rows_before": training_result["train_rows_before"],
        "train_rows_after": training_result["train_rows_after"],
        "train_rows_after_cleaning": training_result.get(
            "train_rows_after_cleaning",
            training_result["train_rows_after"],
        ),
        "train_rows_after_sampling": training_result.get(
            "train_rows_after_sampling",
            training_result["train_rows_after"],
        ),
        "test_rows": training_result["test_rows"],
        "data_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "open_issue_count": int(
            sum(bool(getattr(finding, "found", False)) for finding in findings)
        ),
        "action_log_position": int(action_log_position),
        "class_weight": training_result["class_weight"],
        "sampling": training_result["sampling"],
        "warnings": training_result.get("warnings", []),
        "slice_diagnostics": training_result.get(
            "slice_diagnostics", []
        ),
        "test_indices": training_result.get("test_indices", []),
        "bootstrap_metrics": training_result.get("bootstrap_metrics"),
    }


def _metric_change(before: float, after: float) -> dict[str, float]:
    return {
        "before": float(before),
        "after": float(after),
        "change": float(after - before),
    }


def _bootstrap_change(
    previous: dict[str, Any],
    current: dict[str, Any],
    metric: str,
) -> dict[str, Any] | None:
    previous_bootstrap = previous.get("bootstrap_metrics") or {}
    current_bootstrap = current.get("bootstrap_metrics") or {}
    if (
        previous_bootstrap.get("seed") != current_bootstrap.get("seed")
        or previous.get("test_rows") != current.get("test_rows")
    ):
        return None
    before = np.asarray(previous_bootstrap.get(metric, []), dtype=float)
    after = np.asarray(current_bootstrap.get(metric, []), dtype=float)
    if not len(before) or len(before) != len(after):
        return None
    changes = after - before
    lower, upper = np.quantile(changes, [0.025, 0.975])
    if lower > 0:
        assessment = "larger_than_wobble"
    elif upper < 0:
        assessment = "decline_larger_than_wobble"
    else:
        assessment = "within_wobble"
    return {
        "lower": float(lower),
        "upper": float(upper),
        "assessment": assessment,
        "repetitions": int(len(changes)),
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
            "before": int(
                previous.get(
                    "train_rows_after_cleaning",
                    previous["train_rows_after"],
                )
            ),
            "after": int(
                current.get(
                    "train_rows_after_cleaning",
                    current["train_rows_after"],
                )
            ),
            "change": int(
                current.get(
                    "train_rows_after_cleaning",
                    current["train_rows_after"],
                )
                - previous.get(
                    "train_rows_after_cleaning",
                    previous["train_rows_after"],
                )
            ),
        },
        "training_options": {
            "class_weight": {
                "before": previous.get("class_weight"),
                "after": current.get("class_weight"),
            },
            "sampling": {
                "before": previous.get("sampling"),
                "after": current.get("sampling"),
            },
        },
        "accuracy": _metric_change(
            previous["accuracy"], current["accuracy"]
        ),
        "macro_f1": _metric_change(
            previous["macro_f1"], current["macro_f1"]
        ),
        "roc_auc": None,
        "metric_wobble": {
            "accuracy": _bootstrap_change(previous, current, "accuracy"),
            "macro_f1": _bootstrap_change(previous, current, "macro_f1"),
        },
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
    facts["wobble_caption"] = wobble_caption(facts["metric_wobble"])
    return facts


def wobble_caption(metric_wobble: dict[str, Any]) -> str | None:
    """Explain whether score changes exceed normal test-set variation."""
    accuracy = metric_wobble.get("accuracy")
    if not accuracy:
        return None
    assessment = accuracy["assessment"]
    if assessment == "larger_than_wobble":
        return (
            "The accuracy improvement is larger than normal test-set "
            "wobble in the bootstrap check."
        )
    if assessment == "decline_larger_than_wobble":
        return (
            "The accuracy decline is larger than normal test-set wobble "
            "in the bootstrap check."
        )
    return (
        "The accuracy change is within normal test-set wobble, so it may "
        "not represent a repeatable improvement or decline."
    )


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
    accuracy_wobble = facts.get("metric_wobble", {}).get("accuracy")
    f1_wobble = facts.get("metric_wobble", {}).get("macro_f1")
    accuracy_decline_is_clear = (
        not accuracy_wobble
        or accuracy_wobble["assessment"] == "decline_larger_than_wobble"
    )
    accuracy_gain_is_clear = (
        not accuracy_wobble
        or accuracy_wobble["assessment"] == "larger_than_wobble"
    )
    f1_gain_is_clear = (
        not f1_wobble
        or f1_wobble["assessment"] == "larger_than_wobble"
    )
    if (
        dropped_columns
        and accuracy_change <= -0.05
        and accuracy_decline_is_clear
    ):
        return (
            "The earlier score may have been too optimistic because "
            "accuracy fell after a column was removed."
        )
    if dropped_columns and abs(accuracy_change) < 0.01:
        return "The removed column was not materially helping the score."
    if (
        current.get("class_weight") == "balanced"
        and f1_change > 0.01
        and f1_gain_is_clear
        and abs(accuracy_change) < 0.01
    ):
        return (
            "Balanced class weights improved macro F1 while accuracy "
            "stayed about the same."
        )
    previous_clean_rows = previous.get(
        "train_rows_after_cleaning", previous["train_rows_after"]
    )
    if facts["train_rows"]["change"] < -0.10 * previous_clean_rows:
        if accuracy_change > 0.03 and accuracy_gain_is_clear:
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


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable content fingerprint for report and run provenance."""
    content_hash = pd.util.hash_pandas_object(
        df, index=True
    ).values.tobytes()
    return hashlib.sha256(content_hash).hexdigest()


def build_health_report(
    df: pd.DataFrame,
    target: str,
    findings: list[Any],
    action_log: list[dict[str, Any]],
    run_history: list[dict[str, Any]],
    *,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Build a portable, rule-based record of dataset readiness."""
    finding_rows = [
        finding.to_dict()
        if hasattr(finding, "to_dict")
        else dict(finding)
        for finding in findings
    ]
    unresolved = [row for row in finding_rows if row.get("found")]
    return {
        "report_version": 1,
        "dataset_fingerprint": dataset_fingerprint(df),
        "target": target,
        "split_seed": split_seed,
        "vitals": dataset_vitals(prepare_data(df)),
        "findings": finding_rows,
        "unresolved_warnings": unresolved,
        "actions": action_log,
        "runs": run_history,
    }


def health_report_json(report: dict[str, Any]) -> bytes:
    return json.dumps(
        report,
        indent=2,
        default=str,
        allow_nan=False,
    ).encode("utf-8")

