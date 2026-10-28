import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

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
    }
    assert first["row_count"] == len(df)
    assert set(first["feature_relevance"]) == {
        "numeric",
        "category",
    }
    assert sum(first["feature_relevance"].values()) == pytest.approx(1.0)


def test_imputer_is_fit_on_training_rows_only():
    df = _demo_table()
    df.loc[0:10, "numeric"] = None
    result = train_model(df, "label", seed=42)

    x = df.drop(columns=["label"])
    y = df["label"]
    x_train, _, _, _ = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
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

