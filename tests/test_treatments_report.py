import pandas as pd
import pytest

from data_doctor.report import compare_runs
from data_doctor.treatments import apply_action


def test_action_changes_copy_not_input():
    original = pd.DataFrame(
        {
            "feature": [1.0, None, 3.0],
            "label": ["a", "b", "a"],
        }
    )
    updated, log = apply_action(
        original,
        "label",
        "impute_median",
        column="feature",
    )
    assert original["feature"].isna().sum() == 1
    assert updated["feature"].isna().sum() == 0
    assert log["action"] == "impute_median"


def test_target_cannot_be_dropped():
    df = pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
    try:
        apply_action(df, "label", "drop_column", column="label")
    except ValueError as error:
        assert "target" in str(error).lower()
    else:
        raise AssertionError("Dropping the target should fail.")


def test_run_comparison_reports_facts():
    previous = {
        "action_log_position": 0,
        "columns": ["x", "id"],
        "train_rows_after": 80,
        "accuracy": 0.90,
        "macro_f1": 0.70,
        "roc_auc": 0.85,
    }
    current = {
        "action_log_position": 1,
        "columns": ["x"],
        "train_rows_after": 80,
        "accuracy": 0.89,
        "macro_f1": 0.71,
        "roc_auc": 0.84,
        "class_weight": None,
    }
    actions = [
        {
            "action": "drop_column",
            "column": "id",
        }
    ]
    comparison = compare_runs(previous, current, actions)
    assert comparison["columns_removed"] == ["id"]
    assert comparison["accuracy"]["change"] == pytest.approx(-0.01)
    assert comparison["actions"] == actions

