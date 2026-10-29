"""UI action choices and target-label input helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_doctor.detectors import Finding, MISSING_LABEL_TOKENS


def action_options(
    findings: list[Finding],
    df: pd.DataFrame,
    target: str,
) -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}

    def add(label: str, action: str, column: str | None = None) -> None:
        key = f"{action}|{column or 'dataset'}"
        options[key] = {
            "label": label,
            "action": action,
            "column": column,
        }

    for finding in findings:
        if not finding.found:
            continue
        column = finding.scope if finding.scope != "dataset" else None
        if finding.issue_type == "missing_values" and column:
            if column == target:
                add(
                    f"Drop rows with missing {target}",
                    "drop_rows_missing",
                    column,
                )
            elif finding.evidence["percent"] >= 80:
                add(f"Drop {column}", "drop_column", column)
            else:
                if pd.api.types.is_numeric_dtype(df[column]):
                    add(
                        f"Fill {column} with median",
                        "impute_median",
                        column,
                    )
                    add(
                        f"Fill {column} with mean",
                        "impute_mean",
                        column,
                    )
                else:
                    add(
                        f"Fill {column} with mode",
                        "impute_mode",
                        column,
                    )
                add(
                    f"Fill {column} with a constant",
                    "fill_constant",
                    column,
                )
                add(
                    f"Drop rows missing {column}",
                    "drop_rows_missing",
                    column,
                )
                add(f"Drop {column}", "drop_column", column)
        elif finding.issue_type == "exact_duplicates":
            add("Drop exact duplicate rows", "drop_exact_duplicates")
        elif finding.issue_type == "conflicting_labels":
            add("Drop conflicting-label rows", "drop_conflicting_rows")
        elif finding.issue_type == "outliers" and column:
            add(f"Cap outliers in {column}", "cap_outliers", column)
            add(
                f"Remove outlier rows for {column}",
                "remove_outliers",
                column,
            )
        elif finding.issue_type == "stored_as_text" and column:
            inferred_type = finding.evidence["inferred_type"]
            action = (
                "convert_to_number"
                if inferred_type == "number"
                else "convert_to_datetime"
            )
            add(
                f"Convert {column} to {inferred_type}",
                action,
                column,
            )
        elif finding.issue_type in {
            "id_like",
            "target_copy",
            "suspicious_predictor",
        } and column:
            if column != target:
                add(f"Drop {column}", "drop_column", column)
                add(
                    f"Keep {column} with warning",
                    "keep_with_warning",
                    column,
                )
        elif finding.issue_type == "weird_labels":
            add(
                "Drop missing-like target rows",
                "drop_weird_label_rows",
                target,
            )
            normalized = df[target].astype("string").str.strip().str.lower()
            recodable = (
                normalized.isin(MISSING_LABEL_TOKENS) & df[target].notna()
            )
            if recodable.any():
                add("Recode a target label", "recode_label", target)

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        if column == target:
            continue
        shares = df[column].value_counts(normalize=True, dropna=True)
        if len(shares[shares < 0.01]):
            add(
                f"Group rare {column} values as Other",
                "group_rare_categories",
                column,
            )
    return options


def missing_like_label_values(series: pd.Series) -> list[Any]:
    normalized = series.astype("string").str.strip().str.lower()
    mask = normalized.isin(MISSING_LABEL_TOKENS) & series.notna()
    return series.loc[mask].drop_duplicates().tolist()


def coerce_label_value(raw_value: str, series: pd.Series) -> Any:
    if pd.api.types.is_integer_dtype(series):
        numeric = pd.to_numeric(raw_value, errors="raise")
        if float(numeric).is_integer():
            return int(numeric)
        raise ValueError("The new label must be a whole number.")
    if pd.api.types.is_float_dtype(series):
        return float(pd.to_numeric(raw_value, errors="raise"))
    if pd.api.types.is_bool_dtype(series):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError("The new label must be true or false.")
    return raw_value
