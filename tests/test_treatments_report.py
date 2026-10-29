import pandas as pd
import pytest

from data_doctor.detectors import diagnose
from data_doctor.report import (
    build_health_report,
    compare_runs,
    health_report_json,
)
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


def test_target_label_can_be_explicitly_recoded():
    df = pd.DataFrame(
        {"x": [1, 2, 3], "label": ["yes", "unknown", "no"]}
    )

    updated, log = apply_action(
        df,
        "label",
        "recode_label",
        column="label",
        value={"old": "unknown", "new": "no"},
    )

    assert updated["label"].tolist() == ["yes", "no", "no"]
    assert log["details"] == {"old": "unknown", "new": "no"}


def test_fill_constant_keeps_numeric_type_in_action_log():
    df = pd.DataFrame(
        {"x": pd.Series([1.0, None, 3.0]), "label": ["a", "b", "a"]}
    )
    updated, log = apply_action(
        df, "label", "fill_constant", column="x", value="7.5"
    )
    assert updated["x"].dtype.kind == "f"
    assert log["details"]["fill_value"] == 7.5


def test_stored_text_conversions_are_explicit_and_typed():
    df = pd.DataFrame(
        {
            "amount": ["1.5", " ", "3.0"],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "label": ["a", "b", "a"],
        }
    )
    numeric, numeric_log = apply_action(
        df,
        "label",
        "convert_to_number",
        column="amount",
    )
    dated, date_log = apply_action(
        numeric,
        "label",
        "convert_to_datetime",
        column="event_date",
    )
    assert numeric["amount"].dtype.kind == "f"
    assert numeric["amount"].isna().sum() == 1
    assert pd.api.types.is_datetime64_any_dtype(dated["event_date"])
    assert numeric_log["details"]["converted_to"] == "number"
    assert date_log["details"]["converted_to"] == "datetime"


def test_numeric_text_conversion_rejects_infinity():
    df = pd.DataFrame(
        {
            "amount": ["1.0", "2.0", "inf"],
            "label": ["a", "b", "a"],
        }
    )
    with pytest.raises(ValueError, match="infinity"):
        apply_action(
            df,
            "label",
            "convert_to_number",
            column="amount",
        )


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


def test_run_comparison_labels_bootstrap_wobble_and_improvement():
    base = {
        "action_log_position": 0,
        "columns": ["x"],
        "train_rows_after": 80,
        "accuracy": 0.812,
        "macro_f1": 0.80,
        "roc_auc": None,
        "bootstrap_metrics": {
            "accuracy": [0.80, 0.81, 0.82, 0.81],
            "macro_f1": [0.79, 0.80, 0.81, 0.80],
        },
    }
    within = {
        **base,
        "accuracy": 0.818,
        "bootstrap_metrics": {
            "accuracy": [0.805, 0.805, 0.825, 0.825],
            "macro_f1": [0.795, 0.795, 0.815, 0.815],
        },
    }
    clear = {
        **within,
        "accuracy": 0.86,
        "bootstrap_metrics": {
            "accuracy": [0.85, 0.86, 0.87, 0.86],
            "macro_f1": [0.84, 0.85, 0.86, 0.85],
        },
    }
    within_result = compare_runs(base, within, [])
    clear_result = compare_runs(base, clear, [])
    assert (
        within_result["metric_wobble"]["accuracy"]["assessment"]
        == "within_wobble"
    )
    assert (
        clear_result["metric_wobble"]["accuracy"]["assessment"]
        == "larger_than_wobble"
    )


def test_run_comparison_uses_cleaning_rows_not_oversampling_rows():
    previous = {
        "action_log_position": 0,
        "columns": ["x"],
        "train_rows_after": 80,
        "train_rows_after_cleaning": 80,
        "accuracy": 0.70,
        "macro_f1": 0.60,
        "roc_auc": None,
        "sampling": None,
        "class_weight": None,
    }
    current = {
        **previous,
        "train_rows_after": 140,
        "train_rows_after_sampling": 140,
        "sampling": "oversample",
    }
    comparison = compare_runs(previous, current, [])
    assert comparison["train_rows"]["change"] == 0
    assert comparison["caption"] is None


def test_health_report_is_portable_json():
    df = pd.DataFrame(
        {
            "x": list(range(20)),
            "label": ["a", "b"] * 10,
        }
    )
    report = build_health_report(
        df, "label", diagnose(df, "label"), [], []
    )
    payload = health_report_json(report)
    assert report["dataset_fingerprint"]
    assert b'"target": "label"' in payload

