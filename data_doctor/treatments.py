"""Explicit, one-at-a-time changes to a DataDoctor working copy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from data_doctor.detectors import MISSING_LABEL_TOKENS

ROW_ACTIONS = {
    "drop_exact_duplicates",
    "drop_conflicting_rows",
    "drop_rows_missing",
    "remove_outliers",
    "drop_weird_label_rows",
    "recode_label",
}
LEARNED_ACTIONS = {
    "impute_mean",
    "impute_median",
    "impute_mode",
    "cap_outliers",
    "group_rare_categories",
}
COLUMN_ACTIONS = {"drop_column"}
TYPE_ACTIONS = {"convert_to_number", "convert_to_datetime"}


def action_scope(action: str) -> str:
    """Describe how an action is replayed during honest evaluation."""
    if action in ROW_ACTIONS:
        return "training_rows"
    if action in LEARNED_ACTIONS:
        return "fit_on_train_apply_to_both"
    if (
        action in COLUMN_ACTIONS
        or action in TYPE_ACTIONS
        or action == "fill_constant"
    ):
        return "apply_to_train_and_test"
    if action == "keep_with_warning":
        return "log_only"
    raise ValueError(f"Unsupported action: {action}")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def coerce_constant(value: Any, series: pd.Series) -> Any:
    """Coerce a user constant without silently changing column type."""
    if pd.api.types.is_integer_dtype(series.dtype):
        numeric = pd.to_numeric(value, errors="raise")
        if float(numeric).is_integer():
            return int(numeric)
        raise ValueError("The constant must be a whole number.")
    if pd.api.types.is_float_dtype(series.dtype):
        return float(pd.to_numeric(value, errors="raise"))
    if pd.api.types.is_bool_dtype(series.dtype):
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError("The constant must be true or false.")
    return value


def _convert_stored_text(series: pd.Series, action: str) -> pd.Series:
    normalized = series
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
        series
    ):
        stripped = series.astype("string").str.strip()
        normalized = stripped.mask(stripped.isin({"", "?"}), pd.NA)
    if action == "convert_to_number":
        converted = pd.to_numeric(normalized, errors="raise")
        finite_values = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_values).all():
            raise ValueError(
                "The column contains infinity and cannot be safely "
                "converted to a model-ready number."
            )
        return converted
    if action == "convert_to_datetime":
        return pd.to_datetime(normalized, errors="raise", format="mixed")
    raise ValueError(f"Unsupported type conversion: {action}")


def _entry(
    action: str,
    column: str | None,
    before: pd.DataFrame,
    after: pd.DataFrame,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "column": column,
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "columns_before": int(len(before.columns)),
        "columns_after": int(len(after.columns)),
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _outlier_bounds(series: pd.Series) -> tuple[float, float]:
    values = series.dropna()
    if values.empty or values.nunique() <= 2:
        raise ValueError("This column does not support an IQR treatment.")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _conflicting_mask(df: pd.DataFrame, target: str) -> pd.Series:
    features = [column for column in df.columns if column != target]
    if not features:
        return pd.Series(False, index=df.index)
    target_counts = df.groupby(
        features, dropna=False
    )[target].transform("nunique")
    return target_counts > 1


def apply_action(
    df: pd.DataFrame,
    target: str,
    action: str,
    *,
    column: str | None = None,
    value: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one user-selected action and return a new DataFrame plus log."""
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    if column is not None and column not in df.columns:
        raise ValueError(f"Column {column!r} was not found.")

    before = df
    result = df.copy()
    details: dict[str, Any] = {}

    if action == "drop_exact_duplicates":
        result = result.drop_duplicates().reset_index(drop=True)
    elif action == "drop_conflicting_rows":
        mask = _conflicting_mask(result, target)
        details["removed_rows"] = int(mask.sum())
        result = result.loc[~mask].reset_index(drop=True)
    elif action == "drop_rows_missing":
        if column is None:
            raise ValueError("A column is required.")
        result = result.dropna(subset=[column]).reset_index(drop=True)
    elif action == "drop_column":
        if column is None:
            raise ValueError("A column is required.")
        if column == target:
            raise ValueError("The target column cannot be dropped.")
        result = result.drop(columns=[column])
    elif action in {"impute_mean", "impute_median", "impute_mode"}:
        if column is None:
            raise ValueError("A column is required.")
        if action == "impute_mean":
            fill_value = result[column].mean()
        elif action == "impute_median":
            fill_value = result[column].median()
        else:
            modes = result[column].mode(dropna=True)
            if modes.empty:
                raise ValueError("The column has no value to use as mode.")
            fill_value = modes.iloc[0]
        details["fill_value"] = _python_scalar(fill_value)
        result[column] = result[column].fillna(fill_value)
    elif action == "fill_constant":
        if column is None:
            raise ValueError("A column is required.")
        if value is None:
            raise ValueError("A constant value is required.")
        typed_value = coerce_constant(value, result[column])
        details["fill_value"] = _python_scalar(typed_value)
        result[column] = result[column].fillna(typed_value)
    elif action in TYPE_ACTIONS:
        if column is None:
            raise ValueError("A column is required.")
        result[column] = _convert_stored_text(result[column], action)
        details["converted_to"] = (
            "number" if action == "convert_to_number" else "datetime"
        )
    elif action == "cap_outliers":
        if column is None:
            raise ValueError("A column is required.")
        low, high = _outlier_bounds(result[column])
        details.update({"lower_bound": low, "upper_bound": high})
        result[column] = result[column].clip(lower=low, upper=high)
    elif action == "remove_outliers":
        if column is None:
            raise ValueError("A column is required.")
        low, high = _outlier_bounds(result[column])
        mask = result[column].between(low, high) | result[column].isna()
        details.update(
            {
                "lower_bound": low,
                "upper_bound": high,
                "removed_rows": int((~mask).sum()),
            }
        )
        result = result.loc[mask].reset_index(drop=True)
    elif action == "group_rare_categories":
        if column is None:
            raise ValueError("A column is required.")
        threshold = float(value) if value is not None else 0.01
        shares = result[column].value_counts(
            normalize=True, dropna=True
        )
        rare_values = shares[shares < threshold].index
        details["threshold"] = threshold
        details["grouped_values"] = [str(item) for item in rare_values]
        result[column] = result[column].where(
            ~result[column].isin(rare_values), "Other"
        )
    elif action == "drop_weird_label_rows":
        normalized = (
            result[target].astype("string").str.strip().str.lower()
        )
        mask = (
            normalized.isin(MISSING_LABEL_TOKENS)
            | result[target].isna()
        )
        details["removed_rows"] = int(mask.sum())
        result = result.loc[~mask].reset_index(drop=True)
    elif action == "recode_label":
        if not isinstance(value, dict) or "old" not in value or "new" not in value:
            raise ValueError(
                "Label recoding needs {'old': ..., 'new': ...}."
            )
        result[target] = result[target].replace(
            {value["old"]: value["new"]}
        )
        details.update(value)
    elif action == "keep_with_warning":
        if column is None:
            raise ValueError("A column is required.")
        details["warning_acknowledged"] = True
    else:
        raise ValueError(f"Unsupported action: {action}")

    return result, _entry(action, column, before, result, details)


def apply_actions_to_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target: str,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Replay working-copy actions without learning from test rows.

    Row and label actions affect training rows only. Learned values and
    bounds are fit on training rows and then applied to both feature splits.
    """
    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    for entry in actions:
        action = entry["action"]
        column = entry.get("column")
        details = entry.get("details", {})
        action_scope(action)

        if action == "drop_column" and column in x_train.columns:
            x_train = x_train.drop(columns=[column])
            x_test = x_test.drop(columns=[column])
        elif action == "drop_exact_duplicates":
            table = x_train.copy()
            table[target] = y_train
            keep = ~table.duplicated()
            x_train, y_train = x_train.loc[keep], y_train.loc[keep]
        elif action == "drop_conflicting_rows":
            features = x_train.columns.tolist()
            table = x_train.copy()
            table[target] = y_train
            counts = table.groupby(
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
            if pd.isna(fill_value):
                continue
            x_train[column] = x_train[column].fillna(fill_value)
            x_test[column] = x_test[column].fillna(fill_value)
        elif action == "fill_constant" and column in x_train.columns:
            fill_value = coerce_constant(
                details.get("fill_value"), x_train[column]
            )
            x_train[column] = x_train[column].fillna(fill_value)
            x_test[column] = x_test[column].fillna(fill_value)
        elif action in TYPE_ACTIONS and column in x_train.columns:
            x_train[column] = _convert_stored_text(
                x_train[column], action
            )
            x_test[column] = _convert_stored_text(x_test[column], action)
        elif action in {"cap_outliers", "remove_outliers"}:
            if column not in x_train.columns:
                continue
            try:
                low, high = _outlier_bounds(x_train[column])
            except ValueError:
                continue
            if action == "cap_outliers":
                x_train[column] = x_train[column].clip(low, high)
                x_test[column] = x_test[column].clip(low, high)
            else:
                keep = (
                    x_train[column].between(low, high)
                    | x_train[column].isna()
                )
                x_train, y_train = x_train.loc[keep], y_train.loc[keep]
        elif action == "group_rare_categories":
            if column not in x_train.columns:
                continue
            threshold = float(details.get("threshold", 0.01))
            shares = x_train[column].value_counts(
                normalize=True, dropna=True
            )
            rare_values = shares[shares < threshold].index
            x_train[column] = x_train[column].where(
                ~x_train[column].isin(rare_values), "Other"
            )
            x_test[column] = x_test[column].where(
                ~x_test[column].isin(rare_values), "Other"
            )
        elif action == "drop_weird_label_rows":
            normalized = (
                y_train.astype("string").str.strip().str.lower()
            )
            keep = (
                ~normalized.isin(MISSING_LABEL_TOKENS)
                & y_train.notna()
            )
            x_train, y_train = x_train.loc[keep], y_train.loc[keep]
        elif action == "recode_label":
            y_train = y_train.replace(
                {details.get("old"): details.get("new")}
            )

    return x_train, x_test, y_train, y_test

