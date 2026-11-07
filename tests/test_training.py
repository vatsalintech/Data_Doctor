import numpy as np
import pandas as pd
import pytest

from data_doctor.report import compare_runs
from notebooks.model_training import _slice_diagnostics, train_model


def _demo_table() -> pd.DataFrame:
    rows = 120
    return pd.DataFrame(
        {
            "numeric": [float(i % 17) for i in range(rows)],
            "category": ["red", "blue", "green"] * 40,
            "label": ["yes", "no"] * 60,
        }
    )


def test_training_is_generic_and_deterministic():
    df = _demo_table()
    first = train_model(df, "label", seed=7)
    second = train_model(df, "label", seed=7)
    assert first["metrics"] == second["metrics"]
    assert set(first["metrics"]) >= {
        "accuracy",
        "macro_f1",
        "confusion_matrix",
        "labels",
        "roc_auc",
        "roc_auc_positive_class",
    }
    assert first["row_count"] == len(df)
    assert first["pipeline"].named_steps["model"].solver == "lbfgs"
    assert set(first["feature_relevance"]) == {
        "numeric",
        "category",
    }
    assert sum(first["feature_relevance"].values()) == pytest.approx(1.0)
    assert first["bootstrap_metrics"] == second["bootstrap_metrics"]
    assert len(first["bootstrap_metrics"]["accuracy"]) == 400


def test_imputer_is_fit_on_training_rows_only():
    df = _demo_table()
    df.loc[0:10, "numeric"] = None
    result = train_model(df, "label", seed=42)

    x = df.drop(columns=["label"])
    x_train = x.drop(index=result["test_indices"])
    expected_median = x_train["numeric"].median()
    numeric_pipeline = result["pipeline"].named_steps[
        "preprocessor"
    ].named_transformers_["numeric"]
    fitted_median = numeric_pipeline.named_steps[
        "imputer"
    ].statistics_[0]
    assert fitted_median == expected_median


def test_training_rejects_single_class_target():
    df = _demo_table()
    df["label"] = "one"
    with pytest.raises(ValueError, match="at least two classes"):
        train_model(df, "label")


def test_sampling_changes_training_rows_only():
    df = _demo_table()
    df["label"] = ["rare"] * 20 + ["common"] * 100
    baseline = train_model(df, "label")
    oversampled = train_model(df, "label", sampling="oversample")
    assert baseline["test_rows"] == oversampled["test_rows"]
    assert (
        oversampled["train_rows_after"]
        > oversampled["train_rows_before"]
    )
    assert (
        oversampled["train_rows_after_cleaning"]
        == baseline["train_rows_after_cleaning"]
    )


def test_label_row_action_changes_train_rows_not_test_rows():
    rows = 100
    df = pd.DataFrame(
        {
            "feature": list(range(rows)),
            "label": (
                ["yes", "no"] * 45
                + ["unknown"] * 10
            ),
        }
    )
    result = train_model(
        df,
        "label",
        actions=[{"action": "drop_weird_label_rows"}],
    )

    assert result["train_rows_before"] == 80
    assert result["train_rows_after"] == 72
    assert result["test_rows"] == 20
    assert "unknown" in result["metrics"]["labels"]


def test_train_rows_before_is_counted_before_row_cleaning():
    df = _demo_table()
    df.loc[:19, "numeric"] = None
    result = train_model(
        df,
        "label",
        actions=[
            {
                "action": "drop_rows_missing",
                "column": "numeric",
            }
        ],
    )

    assert result["train_rows_before"] + result["test_rows"] == len(df)
    assert result["train_rows_after"] < result["train_rows_before"]


def test_label_recode_does_not_change_test_labels():
    rows = 100
    df = pd.DataFrame(
        {
            "feature": list(range(rows)),
            "label": (
                ["yes", "no"] * 45
                + ["unknown"] * 10
            ),
        }
    )
    result = train_model(
        df,
        "label",
        actions=[
            {
                "action": "recode_label",
                "details": {"old": "unknown", "new": "no"},
            }
        ],
    )

    model_classes = result["pipeline"].named_steps["model"].classes_
    assert "unknown" not in model_classes
    assert "unknown" in result["metrics"]["labels"]
    assert result["test_rows"] == 20


def test_duplicate_feature_groups_never_cross_train_and_test():
    base = _demo_table().iloc[:60].copy()
    df = pd.concat([base, base], ignore_index=True)
    result = train_model(df, "label")
    test = df.loc[result["test_indices"]].drop(columns=["label"])
    train = df.drop(index=result["test_indices"]).drop(columns=["label"])
    test_hashes = set(pd.util.hash_pandas_object(test, index=False))
    train_hashes = set(pd.util.hash_pandas_object(train, index=False))
    assert test_hashes.isdisjoint(train_hashes)


def test_small_table_has_clear_training_error():
    df = _demo_table().iloc[:8]
    with pytest.raises(ValueError, match="At least 10 rows"):
        train_model(df, "label")


def test_numeric_constant_replay_preserves_numeric_dtype():
    df = _demo_table()
    df.loc[:15, "numeric"] = None
    result = train_model(
        df,
        "label",
        actions=[
            {
                "action": "fill_constant",
                "column": "numeric",
                "details": {"fill_value": 0},
            }
        ],
    )
    numeric = result["pipeline"].named_steps[
        "preprocessor"
    ].named_transformers_["numeric"]
    assert numeric is not None


def test_roc_auc_names_its_positive_class():
    df = _demo_table()
    first = train_model(df, "label", seed=3)
    second = train_model(df, "label", seed=3, class_weight="balanced")
    positive = first["metrics"]["roc_auc_positive_class"]
    model = first["pipeline"].named_steps["model"]
    assert positive == str(model.classes_[1])
    assert second["metrics"]["roc_auc_positive_class"] == positive


def test_conflicting_label_rows_stay_in_the_held_out_split():
    features = list(range(80)) + list(range(20))
    labels = ["yes" if index % 2 == 0 else "no" for index in range(80)]
    labels += ["no" if index % 2 == 0 else "yes" for index in range(20)]
    df = pd.DataFrame({"feature": features, "label": labels})
    preserved = False
    removed_from_train = False
    for seed in range(6):
        baseline = train_model(df, "label", seed=seed)
        cleaned = train_model(
            df,
            "label",
            seed=seed,
            actions=[{"action": "drop_conflicting_rows"}],
        )
        assert cleaned["test_indices"] == baseline["test_indices"]
        test = df.loc[cleaned["test_indices"]]
        counts = test.groupby("feature")["label"].nunique()
        conflicted = counts[counts > 1]
        if not conflicted.empty:
            preserved = True
            feature = conflicted.index[0]
            assert set(
                test.loc[test["feature"].eq(feature), "label"]
            ) == {"yes", "no"}
        if (
            cleaned["train_rows_after_cleaning"]
            < cleaned["train_rows_before"]
        ):
            removed_from_train = True
    assert preserved
    assert removed_from_train


def test_bootstrap_comparison_pairs_the_same_held_out_rows():
    df = _demo_table()
    df.loc[:15, "numeric"] = None
    first = train_model(df, "label", seed=5)
    second = train_model(
        df,
        "label",
        seed=5,
        actions=[{"action": "drop_rows_missing", "column": "numeric"}],
    )
    assert first["test_indices"] == second["test_indices"]

    def record(result: dict, position: int) -> dict:
        return {
            "action_log_position": position,
            "columns": result["columns"],
            "train_rows_after": result["train_rows_after"],
            "train_rows_after_cleaning": result[
                "train_rows_after_cleaning"
            ],
            "test_rows": result["test_rows"],
            "test_indices": result["test_indices"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "roc_auc": result["metrics"]["roc_auc"],
            "roc_auc_positive_class": result["metrics"][
                "roc_auc_positive_class"
            ],
            "bootstrap_metrics": result["bootstrap_metrics"],
            "class_weight": None,
            "sampling": None,
        }

    comparison = compare_runs(
        record(first, 0),
        record(second, 1),
        [{"action": "drop_rows_missing", "column": "numeric"}],
    )
    before = np.asarray(first["bootstrap_metrics"]["accuracy"], dtype=float)
    after = np.asarray(second["bootstrap_metrics"]["accuracy"], dtype=float)
    lower, upper = np.quantile(after - before, [0.025, 0.975])
    assert comparison["held_out_rows_match"] is True
    assert comparison["metric_wobble"]["accuracy"]["lower"] == pytest.approx(
        float(lower)
    )
    assert comparison["metric_wobble"]["accuracy"]["upper"] == pytest.approx(
        float(upper)
    )

    mismatched = record(second, 1)
    mismatched["test_indices"] = list(reversed(mismatched["test_indices"]))
    assert mismatched["test_indices"] != first["test_indices"]
    withheld = compare_runs(record(first, 0), mismatched, [])
    assert withheld["metric_wobble"]["accuracy"] is None
    assert withheld["held_out_rows_match"] is False


def test_slice_diagnostics_guard_small_groups_and_thin_classes():
    features = pd.DataFrame(
        {
            "region": (
                ["north"] * 40 + ["south"] * 20 + ["solo"] * 12 + ["tiny"] * 8
            ),
            "kind": ["mixed"] * 60 + ["solo"] * 20,
            "wide": [f"v{index % 21}" for index in range(80)],
        }
    )
    labels = pd.Series(
        ["yes", "no"] * 20
        + ["yes"] * 19
        + ["no"]
        + ["yes"] * 12
        + ["yes"] * 8
    )
    rows = _slice_diagnostics(features, labels, np.array(["yes"] * 80))
    by_key = {(row["column"], row["value"]): row for row in rows}
    assert ("region", "tiny") not in by_key
    assert all(row["column"] != "wide" for row in rows)

    north = by_key[("region", "north")]
    assert north["macro_f1"] is not None
    assert north["class_count"] == 2
    south = by_key[("region", "south")]
    assert south["support"] >= 10
    assert south["macro_f1"] is None
    assert south["smallest_class_count"] == 1
    assert "fewer than 5" in south["note"]
    solo = by_key[("region", "solo")]
    assert solo["class_count"] == 1
    assert solo["macro_f1"] is None
    assert "Only one target class" in solo["note"]
    assert solo["accuracy"] == 1


def test_feature_relevance_uses_scaled_and_encoded_columns():
    result = train_model(_demo_table(), "label")
    preprocessor = result["pipeline"].named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps[
        "scaler"
    ]
    encoder = preprocessor.named_transformers_["categorical"].named_steps[
        "encoder"
    ]
    assert scaler.mean_.shape == (1,)
    assert scaler.n_features_in_ == 1
    assert list(encoder.categories_[0]) == ["blue", "green", "red"]
    assert set(result["feature_relevance"]) == {"numeric", "category"}
    assert sum(result["feature_relevance"].values()) == pytest.approx(1)


def test_explicit_date_conversion_remains_trainable():
    df = _demo_table()
    df["event_date"] = pd.date_range(
        "2024-01-01",
        periods=len(df),
    ).astype(str)
    result = train_model(
        df,
        "label",
        actions=[
            {
                "action": "convert_to_datetime",
                "column": "event_date",
            }
        ],
    )
    preprocessor = result["pipeline"].named_steps["preprocessor"]
    assert "datetime" in preprocessor.named_transformers_
    assert "event_date" in result["feature_relevance"]

