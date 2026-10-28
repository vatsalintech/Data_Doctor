"""Explicit, one-at-a-time changes to a DataDoctor working copy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from data_doctor.detectors import MISSING_LABEL_TOKENS


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
        details["fill_value"] = str(fill_value)
        result[column] = result[column].fillna(fill_value)
    elif action == "fill_constant":
        if column is None:
            raise ValueError("A column is required.")
        if value is None:
            raise ValueError("A constant value is required.")
        details["fill_value"] = str(value)
        result[column] = result[column].fillna(value)
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
    else:
        raise ValueError(f"Unsupported action: {action}")

    return result, _entry(action, column, before, result, details)

