"""Reusable Logistic Regression training for any classification table."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
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
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from data_doctor.detectors import prepare_data
from data_doctor.treatments import apply_actions_to_split

Sampling = Optional[Literal["oversample", "undersample"]]
BOOTSTRAP_REPETITIONS = 400


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_data(df)


def _validate(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    if len(df.columns) < 2:
        raise ValueError("At least one feature column is required.")
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for training.")
    if df[target].dropna().nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    counts = df[target].dropna().value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Every target class needs at least two rows for splitting."
        )


def _group_aware_split(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Keep identical feature rows together while preserving class balance."""
    groups = pd.util.hash_pandas_object(x, index=False)
    group_table = pd.DataFrame({"group": groups, "target": y})
    # One representative label is enough to stratify each indivisible
    # feature group. Selecting it vectorially avoids a costly Python
    # ``mode`` call for every group (often every row in a large table).
    group_targets = (
        group_table.drop_duplicates("group", keep="first")
        .set_index("group")["target"]
    )
    group_counts = group_targets.value_counts()
    if len(group_targets) < 2 or group_counts.min() < 2:
        raise ValueError(
            "There are not enough independent feature groups in every "
            "class for a leakage-safe train/test split."
        )
    train_groups, test_groups = train_test_split(
        group_targets.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=group_targets.to_numpy(),
    )
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)
    return (
        x.loc[train_mask].copy(),
        x.loc[test_mask].copy(),
        y.loc[train_mask].copy(),
        y.loc[test_mask].copy(),
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


def _datetime_to_days(values: Any) -> np.ndarray:
    """Convert datetime columns to floating-point days since Unix epoch."""
    frame = pd.DataFrame(values)
    converted = np.empty(frame.shape, dtype=float)
    for index, column in enumerate(frame.columns):
        dates = pd.to_datetime(frame[column], errors="coerce")
        numeric = dates.astype("int64").to_numpy(dtype=float)
        numeric[dates.isna().to_numpy()] = np.nan
        converted[:, index] = numeric / 86_400_000_000_000
    return converted


def _build_pipeline(
    x_train: pd.DataFrame,
    *,
    class_weight: str | None,
    seed: int,
) -> Pipeline:
    numeric_columns = x_train.select_dtypes(
        include="number"
    ).columns.tolist()
    datetime_columns = [
        column
        for column in x_train.columns
        if pd.api.types.is_datetime64_any_dtype(x_train[column])
    ]
    categorical_columns = [
        column
        for column in x_train.columns
        if column not in numeric_columns
        and column not in datetime_columns
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
    if datetime_columns:
        transformers.append(
            (
                "datetime",
                Pipeline(
                    steps=[
                        (
                            "to_days",
                            FunctionTransformer(
                                _datetime_to_days,
                                validate=False,
                            ),
                        ),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                datetime_columns,
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
        solver="lbfgs",
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
    coefficients = np.abs(model.coef_)
    encoded_relevance = (
        coefficients.mean(axis=0)
        if coefficients.ndim == 2
        else coefficients
    )

    relevance = {column: 0.0 for column in source_columns}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or not columns:
            continue
        output_slice = preprocessor.output_indices_[name]
        scores = encoded_relevance[output_slice]
        if name in {"numeric", "datetime"}:
            for column, score in zip(columns, scores):
                relevance[column] += float(score)
        elif name == "categorical":
            encoder = transformer.named_steps["encoder"]
            start = 0
            for column, categories in zip(columns, encoder.categories_):
                stop = start + len(categories)
                relevance[column] += float(scores[start:stop].sum())
                start = stop

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


def _slice_diagnostics(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    """Return guarded per-group metrics for categorical test slices."""
    minimum_support = max(10, int(np.ceil(len(x_test) * 0.05)))
    rows: list[dict[str, Any]] = []
    predicted = pd.Series(predictions, index=y_test.index)
    categorical = x_test.select_dtypes(
        include=["object", "string", "category", "bool"]
    )
    for column in categorical.columns:
        if categorical[column].nunique(dropna=True) > 20:
            continue
        grouped_values = categorical[column].astype("string").fillna(
            "(missing)"
        )
        for value, indices in grouped_values.groupby(
            grouped_values
        ).groups.items():
            if len(indices) < minimum_support:
                continue
            truth = y_test.loc[indices]
            guess = predicted.loc[indices]
            rows.append(
                {
                    "column": column,
                    "value": "(missing)" if pd.isna(value) else str(value),
                    "support": int(len(indices)),
                    "accuracy": float(accuracy_score(truth, guess)),
                    "macro_f1": float(
                        f1_score(
                            truth,
                            guess,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
    return rows


def _bootstrap_metric_samples(
    y_test: pd.Series,
    predictions: np.ndarray,
    labels: list[Any],
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    batch_size: int = 20,
) -> dict[str, Any]:
    """Bootstrap test metrics with bounded memory and a repeatable seed."""
    truth = y_test.to_numpy()
    guesses = np.asarray(predictions)
    row_count = len(truth)
    label_codes = {label: index for index, label in enumerate(labels)}
    truth_codes = np.fromiter(
        (label_codes[value] for value in truth),
        dtype=np.int32,
        count=row_count,
    )
    prediction_codes = np.fromiter(
        (label_codes[value] for value in guesses),
        dtype=np.int32,
        count=row_count,
    )
    cell_codes = truth_codes * len(labels) + prediction_codes
    rng = np.random.default_rng(seed)
    accuracy_samples: list[float] = []
    macro_f1_samples: list[float] = []

    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        sampled_rows = rng.integers(
            0,
            row_count,
            size=(size, row_count),
            dtype=np.int32,
        )
        for sampled_cells in cell_codes[sampled_rows]:
            matrix = np.bincount(
                sampled_cells,
                minlength=len(labels) ** 2,
            ).reshape(len(labels), len(labels))
            accuracy_samples.append(float(np.trace(matrix) / row_count))
            actual = matrix.sum(axis=1)
            predicted = matrix.sum(axis=0)
            denominator = actual + predicted
            class_f1 = np.divide(
                2 * np.diag(matrix),
                denominator,
                out=np.zeros(len(labels), dtype=float),
                where=denominator != 0,
            )
            active_classes = int((denominator != 0).sum())
            macro_f1_samples.append(
                float(class_f1.sum() / active_classes)
                if active_classes
                else 0.0
            )

    return {
        "repetitions": repetitions,
        "seed": seed,
        "accuracy": accuracy_samples,
        "macro_f1": macro_f1_samples,
    }


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
    # Missing-like text in the target is a reviewable label, not an
    # automatically missing value. Preserve it until a user action says
    # otherwise. True null targets still cannot be evaluated.
    if target in df.columns:
        prepared[target] = df[target].copy()
    actions = actions or []
    _validate(prepared, target)

    missing_target_rows = int(prepared[target].isna().sum())
    usable = prepared.dropna(subset=[target]).copy()
    x = usable.drop(columns=[target])
    y = usable[target]

    x_train, x_test, y_train, y_test = _group_aware_split(
        x, y, test_size=test_size, seed=seed
    )
    train_rows_before = len(x_train)
    original_test_features = x_test.copy()
    x_train, x_test, y_train, y_test = apply_actions_to_split(
        x_train,
        x_test,
        y_train,
        y_test,
        target,
        actions,
    )
    if y_train.nunique() < 2:
        raise ValueError(
            "The selected training actions leave fewer than two target "
            "classes in the training rows."
        )
    train_rows_after_cleaning = len(x_train)
    x_train, y_train = _resample_training_rows(
        x_train, y_train, sampling, seed
    )
    train_rows_after_sampling = len(x_train)

    pipeline = _build_pipeline(
        x_train, class_weight=class_weight, seed=seed
    )
    training_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x_train, y_train)
    if any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    ):
        training_warnings.append(
            "The model reached its iteration limit; metrics may be unstable."
        )
    predictions = pipeline.predict(x_test)
    model_labels = pipeline.named_steps["model"].classes_
    unseen_test_labels = sorted(
        str(label)
        for label in set(pd.unique(y_test)) - set(model_labels)
    )
    if unseen_test_labels:
        training_warnings.append(
            "The test set contains labels not present after training-row "
            f"actions: {', '.join(unseen_test_labels)}."
        )
    labels = pd.unique(
        pd.concat(
            [
                y_test.reset_index(drop=True),
                pd.Series(predictions),
            ],
            ignore_index=True,
        )
    ).tolist()
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
        "roc_auc_positive_class": None,
    }
    if len(labels) == 2 and len(model_labels) == 2:
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        binary_truth = (y_test == model_labels[1]).astype(int)
        metrics["roc_auc"] = float(
            roc_auc_score(binary_truth, probabilities)
        )
        metrics["roc_auc_positive_class"] = str(model_labels[1])

    bootstrap_metrics = _bootstrap_metric_samples(
        y_test,
        predictions,
        labels,
        seed=seed + 10_000,
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
        "train_rows_after": int(train_rows_after_sampling),
        "train_rows_after_cleaning": int(train_rows_after_cleaning),
        "train_rows_after_sampling": int(train_rows_after_sampling),
        "test_rows": int(len(x_test)),
        "test_indices": x_test.index.tolist(),
        "missing_target_rows_excluded": missing_target_rows,
        "columns": x.columns.tolist(),
        "row_count": int(len(usable)),
        "column_count": int(len(x.columns)),
        "feature_relevance": relevance,
        "warnings": training_warnings,
        "unseen_test_labels": unseen_test_labels,
        "bootstrap_metrics": bootstrap_metrics,
        "slice_diagnostics": _slice_diagnostics(
            original_test_features.loc[x_test.index],
            y_test,
            predictions,
        ),
    }


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    demo = pd.read_csv(
        project_root / "data" / "Messy" / "telco_messy.csv"
    )
    result = train_model(demo, "Churn")
    print(pd.Series(result["metrics"]).to_string())
