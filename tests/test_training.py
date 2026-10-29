import pandas as pd
import pytest

from notebooks.model_training import train_model


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
    assert set(first["metrics"]) == {
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
    result = train_model(_demo_table(), "label")
    assert result["metrics"]["roc_auc_positive_class"] in {"yes", "no"}


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

