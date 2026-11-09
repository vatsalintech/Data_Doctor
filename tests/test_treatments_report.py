import pandas as pd
import pytest

from data_doctor.detectors import diagnose
from data_doctor.report import (
    build_health_report,
    compare_runs,
    health_report_json,
)
from data_doctor.treatments import apply_action, apply_actions_to_split


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


def test_conflicting_labels_are_dropped_from_training_rows_only():
    x_train = pd.DataFrame({"feature": [1, 1, 2, 3, 3]})
    y_train = pd.Series(["yes", "no", "yes", "yes", "yes"])
    x_test = pd.DataFrame({"feature": [4, 4, 5]})
    y_test = pd.Series(["yes", "no", "yes"])
    actions = [{"action": "drop_conflicting_rows"}]

    kept_train, kept_test, kept_y_train, kept_y_test = apply_actions_to_split(
        x_train, x_test, y_train, y_test, "label", actions
    )
    assert list(zip(kept_train["feature"], kept_y_train)) == [
        (2, "yes"),
        (3, "yes"),
        (3, "yes"),
    ]
    assert list(zip(kept_test["feature"], kept_y_test)) == [
        (4, "yes"),
        (4, "no"),
        (5, "yes"),
    ]

    exact_train, exact_test, exact_y_train, exact_y_test = (
        apply_actions_to_split(
            x_train,
            x_test,
            y_train,
            y_test,
            "label",
            [{"action": "drop_exact_duplicates"}],
        )
    )
    assert len(exact_y_train) == len(y_train) - 1
    assert list(zip(exact_test["feature"], exact_y_test)) == list(
        zip(x_test["feature"], y_test)
    )


def test_outlier_caps_use_training_fences_not_logged_or_test_values():
    x_train = pd.DataFrame({"amount": list(range(80))})
    x_test = pd.DataFrame({"amount": [1000.0, 40.0, 41.0, 42.0]})
    y_train = pd.Series(["yes", "no"] * 40)
    y_test = pd.Series(["yes", "no", "yes", "no"])
    _, capped_test, _, _ = apply_actions_to_split(
        x_train,
        x_test,
        y_train,
        y_test,
        "label",
        [
            {
                "action": "cap_outliers",
                "column": "amount",
                "details": {
                    "lower_bound": -10_000,
                    "upper_bound": 10_000,
                },
            }
        ],
    )
    q1, q3 = x_train["amount"].quantile([0.25, 0.75])
    train_high = float(q3 + 1.5 * (q3 - q1))
    full = pd.concat(
        [x_train["amount"], x_test["amount"]], ignore_index=True
    )
    full_q1, full_q3 = full.quantile([0.25, 0.75])
    full_high = float(full_q3 + 1.5 * (full_q3 - full_q1))

    assert capped_test["amount"].iloc[0] == pytest.approx(train_high)
    assert capped_test["amount"].iloc[0] < 1000
    assert capped_test["amount"].iloc[0] != pytest.approx(full_high)


def test_rare_grouping_does_not_change_labels_or_unseen_test_values():
    frame = pd.DataFrame(
        {
            "color": ["red", "blue", "gold"],
            "label": ["yes", "no", "yes"],
        }
    )
    with pytest.raises(ValueError, match="target"):
        apply_action(
            frame,
            "label",
            "group_rare_categories",
            column="label",
        )

    x_train = pd.DataFrame(
        {"color": ["red"] * 50 + ["blue"] * 49 + ["gold"]}
    )
    x_test = pd.DataFrame({"color": ["red", "gold", "silver", "blue"]})
    y_train = pd.Series(["yes", "no"] * 50)
    y_test = pd.Series(["yes", "no", "yes", "no"])
    grouped_train, grouped_test, grouped_y_train, grouped_y_test = (
        apply_actions_to_split(
            x_train,
            x_test,
            y_train,
            y_test,
            "label",
            [
                {
                    "action": "group_rare_categories",
                    "column": "color",
                    "details": {"threshold": 0.02},
                }
            ],
        )
    )
    assert grouped_y_train.tolist() == y_train.tolist()
    assert grouped_y_test.tolist() == y_test.tolist()
    assert grouped_train["color"].tolist().count("Other") == 1
    assert grouped_test["color"].tolist() == [
        "red",
        "Other",
        "silver",
        "blue",
    ]


def test_run_comparison_skips_roc_auc_when_the_positive_class_differs():
    previous = {
        "action_log_position": 0,
        "columns": ["x"],
        "train_rows_after": 80,
        "accuracy": 0.80,
        "macro_f1": 0.70,
        "roc_auc": 0.90,
        "roc_auc_positive_class": "yes",
    }
    changed_class = {
        **previous,
        "roc_auc": 0.40,
        "roc_auc_positive_class": "no",
    }
    same_class = {**previous, "roc_auc": 0.40}
    mismatch = compare_runs(previous, changed_class, [])
    match = compare_runs(previous, same_class, [])
    assert mismatch["roc_auc"] is None
    assert "positive class" in mismatch["roc_auc_note"]
    assert match["roc_auc"]["change"] == pytest.approx(-0.50)


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


def test_assess_feature_target_relationship_leakage_and_id():
    from streamlit_ui.visualizations.target_analysis import assess_feature_target_relationship
    from data_doctor.detectors import Finding

    findings = [
        Finding(
            issue_type="target_copy",
            scope="leaky_col",
            severity="high",
            found=True,
            evidence={},
            suggested_action="drop_column",
            reason="Reveals target.",
        ),
        Finding(
            issue_type="id_like",
            scope="id_col",
            severity="low",
            found=True,
            evidence={},
            suggested_action="drop_column",
            reason="Unique identifier.",
        ),
    ]

    target = pd.Series(["Yes", "No", "Yes", "No"] * 5)
    leaky_col = target.copy()
    id_col = pd.Series(range(20))

    leaky_res = assess_feature_target_relationship(leaky_col, target, findings, "leaky_col", "target")
    assert leaky_res["status"] == "Unsafe (Target Leakage)"
    assert "No — Must Drop" in leaky_res["is_necessary"]

    id_res = assess_feature_target_relationship(id_col, target, findings, "id_col", "target")
    assert id_res["status"] == "Unsafe (Identifier)"
    assert "No — Should Drop" in id_res["is_necessary"]


def test_assess_feature_target_relationship_predictive_and_marginal():
    from streamlit_ui.visualizations.target_analysis import assess_feature_target_relationship

    target = pd.Series(["Yes"] * 20 + ["No"] * 20)
    # Strong categorical feature: Category A has 100% Yes, Category B has 0% Yes
    strong_cat = pd.Series(["A"] * 20 + ["B"] * 20)
    strong_res = assess_feature_target_relationship(strong_cat, target, [], "strong_cat", "target")
    assert strong_res["status"] == "High Importance"
    assert "Critical" in strong_res["is_necessary"]

    # Marginal categorical feature: 50% Yes in both categories
    marginal_cat = pd.Series((["X", "Y"] * 20))
    marginal_res = assess_feature_target_relationship(marginal_cat, target, [], "marginal_cat", "target")
    assert marginal_res["status"] == "Low / Marginal"
    assert "Marginal" in marginal_res["is_necessary"]


def _missing_found(df: pd.DataFrame, target: str, column: str) -> bool:
    findings = diagnose(df, target)
    return any(
        finding.found
        and finding.issue_type == "missing_values"
        and finding.scope == column
        for finding in findings
    )


def test_mode_fill_clears_blank_and_question_mark_cells():
    sex = (["M"] * 12) + (["F"] * 4) + ["?", " ", None, "  ?"]
    df = pd.DataFrame(
        {"Sex": sex, "label": ["yes", "no"] * 10}
    )
    assert _missing_found(df, "label", "Sex")

    updated, log = apply_action(df, "label", "impute_mode", column="Sex")

    assert log["details"]["fill_value"] == "M"
    assert set(updated["Sex"]) == {"M", "F"}
    assert not _missing_found(updated, "label", "Sex")


def test_question_mark_is_not_chosen_as_the_mode():
    df = pd.DataFrame(
        {
            "Sex": ["?"] * 12 + ["F"] * 4 + ["M"] * 2 + [None] * 2,
            "label": ["yes", "no"] * 10,
        }
    )
    updated, log = apply_action(df, "label", "impute_mode", column="Sex")

    assert log["details"]["fill_value"] == "F"
    assert "?" not in set(updated["Sex"].astype(str))
    assert not _missing_found(updated, "label", "Sex")


def test_drop_rows_missing_removes_blank_and_question_mark_rows():
    df = pd.DataFrame(
        {
            "Sex": ["M", "?", " ", None, "F"] + ["M"] * 15,
            "label": ["yes", "no"] * 10,
        }
    )
    updated, _ = apply_action(
        df, "label", "drop_rows_missing", column="Sex"
    )

    assert updated["Sex"].tolist() == ["M", "F"] + ["M"] * 15
    assert not _missing_found(updated, "label", "Sex")


def test_constant_fill_replaces_question_marks():
    df = pd.DataFrame(
        {
            "Sex": ["M", "?", " ", None, "F"] + ["M"] * 15,
            "label": ["yes", "no"] * 10,
        }
    )
    updated, log = apply_action(
        df, "label", "fill_constant", column="Sex", value="F"
    )

    assert log["details"]["fill_value"] == "F"
    assert updated["Sex"].tolist().count("F") == 4
    assert not _missing_found(updated, "label", "Sex")


def test_split_replay_treats_question_marks_as_missing():
    x_train = pd.DataFrame(
        {"Sex": ["M", "M", "F", "?", None, " "]}
    )
    x_test = pd.DataFrame({"Sex": ["?", "F", None]})
    y_train = pd.Series(["yes", "no", "yes", "no", "yes", "no"])
    y_test = pd.Series(["yes", "no", "yes"])
    actions = [{"action": "impute_mode", "column": "Sex"}]

    filled_train, filled_test, filled_y_train, filled_y_test = (
        apply_actions_to_split(
            x_train, x_test, y_train, y_test, "label", actions
        )
    )

    assert filled_train["Sex"].tolist() == ["M", "M", "F", "M", "M", "M"]
    assert filled_test["Sex"].tolist() == ["M", "F", "M"]
    assert filled_y_train.tolist() == y_train.tolist()
    assert filled_y_test.tolist() == y_test.tolist()

    dropped_train, kept_test, dropped_y_train, kept_y_test = (
        apply_actions_to_split(
            x_train,
            x_test,
            y_train,
            y_test,
            "label",
            [{"action": "drop_rows_missing", "column": "Sex"}],
        )
    )
    assert dropped_train["Sex"].tolist() == ["M", "M", "F"]
    assert dropped_y_train.tolist() == ["yes", "no", "yes"]
    assert kept_test["Sex"].tolist() == x_test["Sex"].tolist()
    assert kept_y_test.tolist() == y_test.tolist()


def test_drop_rows_missing_on_target_removes_training_question_marks():
    x_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    x_test = pd.DataFrame({"feature": [5, 6]})
    y_train = pd.Series(["yes", "?", " ", "no"])
    y_test = pd.Series(["?", "no"])

    kept_train, kept_test, kept_y_train, kept_y_test = (
        apply_actions_to_split(
            x_train,
            x_test,
            y_train,
            y_test,
            "label",
            [{"action": "drop_rows_missing", "column": "label"}],
        )
    )

    assert kept_train["feature"].tolist() == [1, 4]
    assert kept_y_train.tolist() == ["yes", "no"]
    assert kept_test["feature"].tolist() == [5, 6]
    assert kept_y_test.tolist() == ["?", "no"]

