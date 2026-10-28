"""Reusable Logistic Regression training for any classification table."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Sampling = Optional[Literal["oversample", "undersample"]]


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    text_columns = prepared.select_dtypes(
        include=["object", "string"]
    ).columns
    for column in text_columns:
        stripped = prepared[column].astype("string").str.strip()
        prepared[column] = prepared[column].mask(
            stripped.isin({"", "?"}), pd.NA
        )
    return prepared


def _validate(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    if len(df.columns) < 2:
        raise ValueError("At least one feature column is required.")
    if df[target].dropna().nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    counts = df[target].dropna().value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Every target class needs at least two rows for splitting."
        )


def _resample_training_rows(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sampling: Sampling,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sampling is None:
        return x_train, y_train
    if sampling not in {"oversample", "undersample"}:
        raise ValueError(
            "sampling must be None, 'oversample', or 'undersample'."
        )

    rng = np.random.default_rng(seed)
    grouped_indices = {
        label: y_train.index[y_train == label].to_numpy()
        for label in y_train.unique()
    }
    desired = (
        max(len(indices) for indices in grouped_indices.values())
        if sampling == "oversample"
        else min(len(indices) for indices in grouped_indices.values())
    )

    sampled_indices = []
    for indices in grouped_indices.values():
        sampled_indices.extend(
            rng.choice(
                indices,
                size=desired,
                replace=sampling == "oversample",
            ).tolist()
        )
    rng.shuffle(sampled_indices)
    return (
        x_train.loc[sampled_indices].reset_index(drop=True),
        y_train.loc[sampled_indices].reset_index(drop=True),
    )


def _apply_actions_to_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target: str,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Replay cleaning honestly: learned values use train rows only."""
    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    for entry in actions:
        action = entry["action"]
        column = entry.get("column")
        details = entry.get("details", {})

        if action == "drop_column" and column in x_train.columns:
            x_train = x_train.drop(columns=[column])
            x_test = x_test.drop(columns=[column])
        elif action == "drop_exact_duplicates":
            train_table = x_train.copy()
            train_table[target] = y_train
            keep = ~train_table.duplicated()
            x_train, y_train = x_train.loc[keep], y_train.loc[keep]
        elif action == "drop_conflicting_rows":
            features = x_train.columns.tolist()
            train_table = x_train.copy()
            train_table[target] = y_train
            counts = train_table.groupby(
                features, dropna=False
            )[target].transform("nunique")
            keep = counts <= 1
            x_train, y_train = x_train.loc[keep], y_train.loc[keep]
        elif action == "drop_rows_missing" and column in x_train.columns:
            keep = x_train[column].notna()
            x_train, y_train = x_train.loc[keep], y_train.loc[keep]
        elif action in {
            "impute_mean",
            "impute_median",
            "impute_mode",
        } and column in x_train.columns:
            if action == "impute_mean":
                fill_value = x_train[column].mean()
            elif action == "impute_median":
                fill_value = x_train[column].median()
            else:
                modes = x_train[column].mode(dropna=True)
                if modes.empty:
                    continue
                fill_value = modes.iloc[0]
            x_train[column] = x_train[column].fillna(fill_value)
            x_test[column] = x_test[column].fillna(fill_value)
        elif action == "fill_constant" and column in x_train.columns:
            fill_value = details.get("fill_value")
            x_train[column] = x_train[column].fillna(fill_value)
            x_test[column] = x_test[column].fillna(fill_value)
        elif action in {"cap_outliers", "remove_outliers"}:
            if column not in x_train.columns:
                continue
            values = x_train[column].dropna()
            if values.nunique() <= 2:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            if action == "cap_outliers":
                x_train[column] = x_train[column].clip(low, high)
                x_test[column] = x_test[column].clip(low, high)
            else:
                keep = (
                    x_train[column].between(low, high)
                    | x_train[column].isna()
                )
                x_train, y_train = (
                    x_train.loc[keep],
                    y_train.loc[keep],
                )
        elif action == "group_rare_categories":
            if column not in x_train.columns:
                continue
            shares = x_train[column].value_counts(
                normalize=True, dropna=True
            )
            rare_values = shares[shares < 0.01].index
            x_train[column] = x_train[column].where(
                ~x_train[column].isin(rare_values), "Other"
            )
            x_test[column] = x_test[column].where(
                ~x_test[column].isin(rare_values), "Other"
            )

    return x_train, x_test, y_train, y_test


def _build_pipeline(
    x_train: pd.DataFrame,
    *,
    class_weight: str | None,
    seed: int,
) -> Pipeline:
    numeric_columns = x_train.select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_columns = [
        column
        for column in x_train.columns
        if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns remain.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        random_state=seed,
        solver="saga",
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )


def _feature_relevance(
    pipeline: Pipeline,
    source_columns: list[str],
) -> dict[str, float]:
    """Aggregate absolute encoded coefficients back to source columns."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.abs(model.coef_)
    encoded_relevance = (
        coefficients.mean(axis=0)
        if coefficients.ndim == 2
        else coefficients
    )

    relevance = {column: 0.0 for column in source_columns}
    longest_first = sorted(source_columns, key=len, reverse=True)
    for encoded_name, importance in zip(
        feature_names, encoded_relevance
    ):
        transformed_name = encoded_name.split("__", 1)[-1]
        source = next(
            (
                column
                for column in longest_first
                if transformed_name == column
                or transformed_name.startswith(f"{column}_")
            ),
            None,
        )
        if source is not None:
            relevance[source] += float(importance)

    total = sum(relevance.values())
    if total > 0:
        relevance = {
            column: score / total
            for column, score in relevance.items()
        }
    return dict(
        sorted(
            relevance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def train_model(
    df: pd.DataFrame,
    target: str,
    *,
    seed: int = 42,
    test_size: float = 0.20,
    class_weight: str | None = None,
    sampling: Sampling = None,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Split, preprocess, train, and evaluate Logistic Regression.

    Learned preprocessing is fit on training rows only. The input DataFrame
    is never mutated.
    """
    prepared = _prepare_table(df)
    actions = actions or []
    for entry in actions:
        if entry["action"] == "recode_label":
            details = entry.get("details", {})
            prepared[target] = prepared[target].replace(
                {details.get("old"): details.get("new")}
            )
        elif entry["action"] == "drop_weird_label_rows":
            normalized = (
                prepared[target]
                .astype("string")
                .str.strip()
                .str.lower()
            )
            weird = normalized.isin(
                {
                    "",
                    "?",
                    "???",
                    "-1",
                    "na",
                    "n/a",
                    "none",
                    "null",
                    "unknown",
                }
            ) | prepared[target].isna()
            prepared = prepared.loc[~weird].copy()
    _validate(prepared, target)

    missing_target_rows = int(prepared[target].isna().sum())
    usable = prepared.dropna(subset=[target]).copy()
    x = usable.drop(columns=[target])
    y = usable[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    x_train, x_test, y_train, y_test = _apply_actions_to_split(
        x_train,
        x_test,
        y_train,
        y_test,
        target,
        actions,
    )
    train_rows_before = len(x_train)
    x_train, y_train = _resample_training_rows(
        x_train, y_train, sampling, seed
    )

    pipeline = _build_pipeline(
        x_train, class_weight=class_weight, seed=seed
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    labels = pipeline.named_steps["model"].classes_
    relevance = _feature_relevance(
        pipeline, x_train.columns.tolist()
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "labels": [str(label) for label in labels],
        "roc_auc": None,
    }
    if len(labels) == 2:
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        binary_truth = (y_test == labels[1]).astype(int)
        metrics["roc_auc"] = float(
            roc_auc_score(binary_truth, probabilities)
        )

    return {
        "metrics": metrics,
        "pipeline": pipeline,
        "target": target,
        "seed": seed,
        "test_size": test_size,
        "class_weight": class_weight,
        "sampling": sampling,
        "train_rows_before": int(train_rows_before),
        "train_rows_after": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "missing_target_rows_excluded": missing_target_rows,
        "columns": x.columns.tolist(),
        "row_count": int(len(usable)),
        "column_count": int(len(x.columns)),
        "feature_relevance": relevance,
    }


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    demo = pd.read_csv(
        project_root / "data" / "Messy" / "telco_messy.csv"
    )
    result = train_model(demo, "Churn")
    print(pd.Series(result["metrics"]).to_string())
