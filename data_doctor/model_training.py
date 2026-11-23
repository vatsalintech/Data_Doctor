"""Leakage-safe model training pipeline (Logistic Regression and HistGradientBoostingClassifier)."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)

from data_doctor.detectors import prepare_data
from data_doctor.treatments import apply_actions_to_split

Sampling = Optional[Literal["oversample", "undersample"]]
ModelType = Literal["logistic_regression", "hist_gradient_boosting"]
BOOTSTRAP_REPETITIONS = 400


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_data(df)


def _sklearn_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn pandas missing markers into np.nan for scikit-learn.

    Text cleaning stores blanks as ``pd.NA``. SimpleImputer cannot test
    that marker as a boolean, so training converts its own copies only.
    """
    converted = frame.copy()
    for column in converted.columns:
        series = converted[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            if pd.api.types.is_extension_array_dtype(series.dtype):
                converted[column] = series.astype("float64")
            continue
        if (
            series.dtype == object
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            values = series.astype(object)
            converted[column] = values.where(values.notna(), np.nan)
    return converted


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


def _group_aware_cv_splits(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/val positional index splits keeping identical feature rows together."""
    groups = pd.util.hash_pandas_object(x, index=False)
    group_table = pd.DataFrame({"group": groups, "target": y})
    group_targets = (
        group_table.drop_duplicates("group", keep="first")
        .set_index("group")["target"]
    )
    group_counts = group_targets.value_counts()
    min_count = int(group_counts.min()) if not group_counts.empty else 0
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        raise ValueError(
            f"Cannot perform {n_splits}-fold cross-validation: the smallest target class "
            f"has only {min_count} independent group(s). At least 2 groups are required."
        )

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    unique_groups = group_targets.index.to_numpy()
    group_labels = group_targets.to_numpy()

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    groups_array = groups.to_numpy()
    for train_grp_idx, val_grp_idx in skf.split(unique_groups, group_labels):
        train_groups_set = set(unique_groups[train_grp_idx])
        val_groups_set = set(unique_groups[val_grp_idx])
        train_indices = np.where(np.isin(groups_array, list(train_groups_set)))[0]
        val_indices = np.where(np.isin(groups_array, list(val_groups_set)))[0]
        splits.append((train_indices, val_indices))
    return splits


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
    usable_numeric = [
        column
        for column in numeric_columns
        if not x_train[column].dropna().empty
    ]
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
    if usable_numeric:
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
                usable_numeric,
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
                                max_categories=50,
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


def _build_hgb_pipeline(
    x_train: pd.DataFrame,
    *,
    class_weight: str | None,
    seed: int,
) -> Pipeline:
    """Build a HistGradientBoostingClassifier pipeline.

    HGB natively handles missing values in numeric columns and accepts
    categorical features directly via OrdinalEncoder with unknown handling.
    Datetime columns are converted to numeric days first.
    """
    numeric_columns = x_train.select_dtypes(
        include="number"
    ).columns.tolist()
    usable_numeric = [
        column
        for column in numeric_columns
        if not x_train[column].dropna().empty
    ]
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
    if usable_numeric:
        # HGB handles NaN natively — no imputation needed for numerics.
        transformers.append(("numeric", "passthrough", usable_numeric))
    if datetime_columns:
        # Convert datetimes to numeric days; HGB will handle any remaining NaN.
        transformers.append(
            (
                "datetime",
                FunctionTransformer(_datetime_to_days, validate=False),
                datetime_columns,
            )
        )
    if categorical_columns:
        # OrdinalEncoder is required; HGB reads integer codes directly.
        transformers.append(
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    max_categories=50,
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns remain.")

    # Build the list of categorical feature indices for HGB so it treats
    # them as unordered categories rather than numeric magnitudes.
    # The preprocessor outputs: numerics first, then datetimes, then categoricals.
    n_numeric = len(usable_numeric)
    n_datetime = len(datetime_columns)
    n_categorical = len(categorical_columns)
    categorical_features = list(
        range(n_numeric + n_datetime, n_numeric + n_datetime + n_categorical)
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    model = HistGradientBoostingClassifier(
        max_iter=300,
        class_weight=class_weight,
        random_state=seed,
        categorical_features=categorical_features if categorical_features else None,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )


def _feature_relevance(
    pipeline: Pipeline,
    source_columns: list[str],
    *,
    x_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Aggregate model importance scores back to source columns.

    For Logistic Regression the absolute coefficient weights are used.
    For HistGradientBoostingClassifier, permutation feature importance is
    computed on held-out evaluation rows (measuring drop in prediction accuracy).
    Both are normalised to sum to 1.0.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        # Non-linear model (e.g. HistGradientBoostingClassifier):
        # Compute permutation feature importance on evaluation rows.
        if (
            x_eval is not None
            and y_eval is not None
            and len(x_eval) > 0
            and len(source_columns) > 0
        ):
            sample_size = min(len(x_eval), 400)
            if len(x_eval) > sample_size:
                eval_idx = np.random.default_rng(seed).choice(
                    len(x_eval), size=sample_size, replace=False
                )
                x_sample = x_eval.iloc[eval_idx]
                y_sample = y_eval.iloc[eval_idx]
            else:
                x_sample = x_eval
                y_sample = y_eval

            try:
                perm_result = permutation_importance(
                    pipeline,
                    x_sample,
                    y_sample,
                    n_repeats=3,
                    random_state=seed,
                    scoring="accuracy",
                    n_jobs=1,
                )
                mean_scores = np.maximum(0.0, perm_result.importances_mean)
                total = float(mean_scores.sum())
                if total > 0:
                    relevance = {
                        col: float(score / total)
                        for col, score in zip(x_sample.columns, mean_scores)
                    }
                    # Ensure all source columns exist in relevance dict
                    for col in source_columns:
                        if col not in relevance:
                            relevance[col] = 0.0
                    return dict(
                        sorted(
                            relevance.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )
            except Exception:
                pass  # Fall back to equal weights if permutation fails

        n = len(source_columns)
        if n == 0:
            return {}
        equal = 1.0 / n
        return {col: equal for col in source_columns}

    # Logistic Regression: average absolute coefficients across classes
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
            n_outs = getattr(
                encoder,
                "_n_features_outs",
                [len(c) for c in encoder.categories_],
            )
            start = 0
            for column, n_features in zip(columns, n_outs):
                stop = start + n_features
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
    """Return guarded per-group metrics for categorical test slices.

    A group needs enough rows to be shown. Macro F1 is included only when
    every target class also has enough rows inside that group. A large
    group that contains only one class still reports accuracy.
    """
    minimum_support = max(10, int(np.ceil(len(x_test) * 0.05)))
    minimum_class_rows = 5
    rows: list[dict[str, Any]] = []
    predicted = pd.Series(predictions, index=y_test.index)
    required_classes = pd.Series(y_test).dropna().value_counts()
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
            class_counts = truth.value_counts(dropna=True)
            aligned = class_counts.reindex(
                required_classes.index, fill_value=0
            )
            class_count = int((class_counts > 0).sum())
            smallest_class_count = (
                int(class_counts.min()) if class_count else 0
            )
            macro_f1 = None
            note = None
            classes_are_represented = (
                len(required_classes) >= 2
                and int(aligned.min()) >= minimum_class_rows
            )
            if classes_are_represented:
                macro_f1 = float(
                    f1_score(
                        truth,
                        guess,
                        average="macro",
                        labels=list(required_classes.index),
                        zero_division=0,
                    )
                )
            elif class_count < 2:
                note = (
                    "Only one target class appears here, so macro F1 "
                    "is omitted."
                )
            else:
                note = (
                    "A target class has fewer than 5 rows here, so "
                    "macro F1 is omitted."
                )
            rows.append(
                {
                    "column": column,
                    "value": "(missing)" if pd.isna(value) else str(value),
                    "support": int(len(indices)),
                    "class_count": class_count,
                    "smallest_class_count": smallest_class_count,
                    "accuracy": float(accuracy_score(truth, guess)),
                    "macro_f1": macro_f1,
                    "note": note or "",
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


def _train_model_cv(
    prepared: pd.DataFrame,
    target: str,
    *,
    cv_folds: int = 5,
    seed: int = 42,
    class_weight: str | None = None,
    sampling: Sampling = None,
    actions: list[dict[str, Any]] | None = None,
    positive_class: str | None = None,
    model_type: ModelType = "logistic_regression",
) -> dict[str, Any]:
    """Train and evaluate with group-aware stratified K-Fold cross-validation."""
    actions = actions or []
    missing_target_rows = int(prepared[target].isna().sum())
    usable = prepared.dropna(subset=[target]).copy()
    x = usable.drop(columns=[target])
    y = usable[target]

    splits = _group_aware_cv_splits(x, y, n_splits=cv_folds, seed=seed)
    actual_folds = len(splits)

    unique_classes = sorted(list(pd.unique(y)))
    is_binary = len(unique_classes) == 2
    pos_class: str | None = None
    if is_binary:
        sorted_labels = [str(c) for c in unique_classes]
        if positive_class and str(positive_class) in sorted_labels:
            pos_class = str(positive_class)
        else:
            pos_class = sorted_labels[1]

    oof_preds = np.empty(len(usable), dtype=object)
    oof_probs = np.zeros((len(usable), len(unique_classes)), dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    training_warnings: list[str] = []

    for fold_num, (tr_idx, val_idx) in enumerate(splits, start=1):
        x_tr = x.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        x_val = x.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()

        x_tr, x_val, y_tr, y_val = apply_actions_to_split(
            x_tr, x_val, y_tr, y_val, target, actions
        )
        if y_tr.nunique() < 2:
            continue

        x_tr, y_tr = _resample_training_rows(
            x_tr, y_tr, sampling, seed + fold_num
        )
        x_tr = _sklearn_features(x_tr)
        x_val = _sklearn_features(x_val)

        if model_type == "hist_gradient_boosting":
            fold_pipe = _build_hgb_pipeline(
                x_tr, class_weight=class_weight, seed=seed + fold_num
            )
        else:
            fold_pipe = _build_pipeline(
                x_tr, class_weight=class_weight, seed=seed + fold_num
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fold_pipe.fit(x_tr, y_tr)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            warn_msg = "The model reached its iteration limit in cross-validation folds."
            if warn_msg not in training_warnings:
                training_warnings.append(warn_msg)

        preds = fold_pipe.predict(x_val)
        probs = fold_pipe.predict_proba(x_val)
        model_classes = list(fold_pipe.named_steps["model"].classes_)

        for cls_i, cls_val in enumerate(model_classes):
            if cls_val in unique_classes:
                target_cls_idx = unique_classes.index(cls_val)
                oof_probs[val_idx, target_cls_idx] = probs[:, cls_i]

        oof_preds[val_idx] = preds

        f_acc = float(accuracy_score(y_val, preds))
        f_macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        f_weighted_f1 = float(f1_score(y_val, preds, average="weighted", zero_division=0))
        f_macro_p = float(precision_score(y_val, preds, average="macro", zero_division=0))
        f_macro_r = float(recall_score(y_val, preds, average="macro", zero_division=0))
        f_roc_auc = None
        f_ap = None

        if is_binary and pos_class in [str(c) for c in model_classes]:
            pos_idx = [str(c) for c in model_classes].index(pos_class)
            val_bin = (y_val.astype(str) == pos_class).astype(int)
            if val_bin.sum() > 0 and (1 - val_bin).sum() > 0:
                try:
                    f_roc_auc = float(roc_auc_score(val_bin, probs[:, pos_idx]))
                    f_ap = float(average_precision_score(val_bin, probs[:, pos_idx]))
                except Exception:
                    pass

        fold_metrics.append({
            "fold": fold_num,
            "train_rows": int(len(x_tr)),
            "val_rows": int(len(x_val)),
            "accuracy": f_acc,
            "macro_f1": f_macro_f1,
            "macro_precision": f_macro_p,
            "macro_recall": f_macro_r,
            "weighted_f1": f_weighted_f1,
            "roc_auc": f_roc_auc,
            "average_precision": f_ap,
        })

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    precs = [m["macro_precision"] for m in fold_metrics]
    recs = [m["macro_recall"] for m in fold_metrics]
    wt_f1s = [m["weighted_f1"] for m in fold_metrics]
    rocs = [m["roc_auc"] for m in fold_metrics if m["roc_auc"] is not None]
    aps = [m["average_precision"] for m in fold_metrics if m["average_precision"] is not None]

    cv_summary = {
        "folds": actual_folds,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "macro_precision_mean": float(np.mean(precs)),
        "macro_precision_std": float(np.std(precs)),
        "macro_recall_mean": float(np.mean(recs)),
        "macro_recall_std": float(np.std(recs)),
        "weighted_f1_mean": float(np.mean(wt_f1s)),
        "weighted_f1_std": float(np.std(wt_f1s)),
        "roc_auc_mean": float(np.mean(rocs)) if rocs else None,
        "roc_auc_std": float(np.std(rocs)) if rocs else None,
        "average_precision_mean": float(np.mean(aps)) if aps else None,
        "average_precision_std": float(np.std(aps)) if aps else None,
    }

    # Train a final pipeline on the full dataset so the user has an exportable model
    x_full_clean, _, y_full_clean, _ = apply_actions_to_split(
        x, x.iloc[:0], y, y.iloc[:0], target, actions
    )
    x_full_clean, y_full_clean = _resample_training_rows(
        x_full_clean, y_full_clean, sampling, seed
    )
    x_full_clean_features = _sklearn_features(x_full_clean)
    if model_type == "hist_gradient_boosting":
        final_pipeline = _build_hgb_pipeline(
            x_full_clean_features, class_weight=class_weight, seed=seed
        )
    else:
        final_pipeline = _build_pipeline(
            x_full_clean_features, class_weight=class_weight, seed=seed
        )
    final_pipeline.fit(x_full_clean_features, y_full_clean)

    relevance = _feature_relevance(
        final_pipeline,
        x.columns.tolist(),
        x_eval=x_full_clean_features,
        y_eval=y_full_clean,
        seed=seed,
    )

    valid_oof = [i for i, p in enumerate(oof_preds) if p is not None]
    y_oof = y.iloc[valid_oof]
    p_oof = oof_preds[valid_oof]
    probs_oof = oof_probs[valid_oof]

    labels = pd.unique(
        pd.concat([y_oof.reset_index(drop=True), pd.Series(p_oof)], ignore_index=True)
    ).tolist()

    metrics: dict[str, Any] = {
        "accuracy": cv_summary["accuracy_mean"],
        "macro_f1": cv_summary["macro_f1_mean"],
        "macro_precision": cv_summary["macro_precision_mean"],
        "macro_recall": cv_summary["macro_recall_mean"],
        "weighted_precision": float(precision_score(y_oof, p_oof, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_oof, p_oof, average="weighted", zero_division=0)),
        "weighted_f1": cv_summary["weighted_f1_mean"],
        "confusion_matrix": confusion_matrix(y_oof, p_oof, labels=labels).tolist(),
        "labels": [str(l) for l in labels],
        "roc_auc": cv_summary["roc_auc_mean"],
        "roc_auc_positive_class": pos_class,
        "average_precision": cv_summary["average_precision_mean"],
        "positive_class": pos_class,
        "per_class_metrics": [],
    }

    model_labels = final_pipeline.named_steps["model"].classes_
    class_precisions = precision_score(
        y_oof, p_oof, labels=list(model_labels), average=None, zero_division=0
    )
    class_recalls = recall_score(
        y_oof, p_oof, labels=list(model_labels), average=None, zero_division=0
    )
    class_f1s = f1_score(
        y_oof, p_oof, labels=list(model_labels), average=None, zero_division=0
    )
    class_supports = [int((y_oof == cls).sum()) for cls in model_labels]
    for cls, prec, rec, f1_val, sup in zip(
        model_labels, class_precisions, class_recalls, class_f1s, class_supports
    ):
        metrics["per_class_metrics"].append({
            "class": str(cls),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_val),
            "support": sup,
        })

    # Fold mean ± std is the stability signal for CV. The paired bootstrap
    # check is reserved for the hold-out path, where two runs share one test set.
    return {
        "metrics": metrics,
        "pipeline": final_pipeline,
        "target": target,
        "seed": seed,
        "test_size": 1.0 / actual_folds,
        "class_weight": class_weight,
        "sampling": sampling,
        "positive_class": positive_class,
        "model_type": model_type,
        "train_rows_before": int(len(usable)),
        "train_rows_after": int(len(x_full_clean)),
        "train_rows_after_cleaning": int(len(x_full_clean)),
        "train_rows_after_sampling": int(len(x_full_clean)),
        "test_rows": int(len(y_oof)),
        "test_indices": usable.index[valid_oof].tolist(),
        "test_probabilities": probs_oof.tolist(),
        "test_prob_labels": [str(c) for c in unique_classes],
        "y_test": y_oof.tolist(),
        "missing_target_rows_excluded": missing_target_rows,
        "columns": x_full_clean.columns.tolist(),
        "row_count": int(len(usable)),
        "column_count": int(len(x_full_clean.columns)),
        "feature_relevance": relevance,
        "warnings": training_warnings,
        "unseen_test_labels": [],
        "bootstrap_metrics": None,
        "cv_folds": actual_folds,
        "cv_summary": cv_summary,
        "cv_fold_metrics": fold_metrics,
        "slice_diagnostics": _slice_diagnostics(
            x.iloc[valid_oof],
            y_oof,
            p_oof,
        ),
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
    positive_class: str | None = None,
    model_type: ModelType = "logistic_regression",
    cv_folds: int | None = None,
) -> dict[str, Any]:
    """Split, preprocess, train, and evaluate a classification model.

    Supported model_type values:
    - ``"logistic_regression"``  – sklearn LogisticRegression (default)
    - ``"hist_gradient_boosting"`` – sklearn HistGradientBoostingClassifier

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

    if cv_folds is not None and cv_folds >= 2:
        return _train_model_cv(
            prepared,
            target,
            cv_folds=cv_folds,
            seed=seed,
            class_weight=class_weight,
            sampling=sampling,
            actions=actions,
            positive_class=positive_class,
            model_type=model_type,
        )

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
    x_train = _sklearn_features(x_train)
    x_test = _sklearn_features(x_test)

    if model_type == "hist_gradient_boosting":
        pipeline = _build_hgb_pipeline(
            x_train, class_weight=class_weight, seed=seed
        )
    else:
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
        pipeline,
        x_train.columns.tolist(),
        x_eval=x_test,
        y_eval=y_test,
        seed=seed,
    )

    # Save probabilities for threshold-free metrics and UI threshold slider
    test_probabilities: np.ndarray = pipeline.predict_proba(x_test)
    test_prob_labels: list[str] = [str(c) for c in model_labels]

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "macro_precision": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "labels": [str(label) for label in labels],
        "roc_auc": None,
        "roc_auc_positive_class": None,
        "average_precision": None,
        "positive_class": None,
        "per_class_metrics": [],
    }

    # Per-class precision, recall, F1
    per_class: list[dict[str, Any]] = []
    class_precisions = precision_score(
        y_test, predictions, labels=list(model_labels), average=None, zero_division=0
    )
    class_recalls = recall_score(
        y_test, predictions, labels=list(model_labels), average=None, zero_division=0
    )
    class_f1s = f1_score(
        y_test, predictions, labels=list(model_labels), average=None, zero_division=0
    )
    class_supports = [
        int((y_test == cls).sum()) for cls in model_labels
    ]
    for cls, prec, rec, f1_val, sup in zip(
        model_labels, class_precisions, class_recalls, class_f1s, class_supports
    ):
        per_class.append({
            "class": str(cls),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_val),
            "support": sup,
        })
    metrics["per_class_metrics"] = per_class

    # Binary-only metrics: ROC-AUC, PR-AUC, positive-class Precision/Recall/F1
    is_binary = len(labels) == 2 and len(model_labels) == 2
    if is_binary:
        # Resolve positive class: user choice → second sorted class as fallback
        sorted_model_labels = [str(c) for c in model_labels]
        if positive_class and str(positive_class) in sorted_model_labels:
            pos_class = str(positive_class)
        else:
            pos_class = sorted_model_labels[1]
        metrics["positive_class"] = pos_class
        pos_idx = sorted_model_labels.index(pos_class)
        pos_probabilities = test_probabilities[:, pos_idx]
        binary_truth = (y_test.astype(str) == pos_class).astype(int)
        if binary_truth.sum() > 0:
            metrics["roc_auc"] = float(
                roc_auc_score(binary_truth, pos_probabilities)
            )
            metrics["average_precision"] = float(
                average_precision_score(binary_truth, pos_probabilities)
            )
        metrics["roc_auc_positive_class"] = pos_class

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
        "positive_class": positive_class,
        "model_type": model_type,
        "train_rows_before": int(train_rows_before),
        "train_rows_after": int(train_rows_after_sampling),
        "train_rows_after_cleaning": int(train_rows_after_cleaning),
        "train_rows_after_sampling": int(train_rows_after_sampling),
        "test_rows": int(len(x_test)),
        "test_indices": x_test.index.tolist(),
        "test_probabilities": test_probabilities.tolist(),
        "test_prob_labels": test_prob_labels,
        "y_test": y_test.tolist(),
        "missing_target_rows_excluded": missing_target_rows,
        "columns": x_train.columns.tolist(),
        "row_count": int(len(usable)),
        "column_count": int(len(x_train.columns)),
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
