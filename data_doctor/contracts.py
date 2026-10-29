"""Portable schema contracts and rule-based dataset drift checks."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def infer_contract(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Infer a conservative validation contract from the working table."""
    columns: dict[str, Any] = {}
    for name in df.columns:
        series = df[name]
        family = _dtype_family(series)
        missing_rate = float(series.isna().mean())
        rule: dict[str, Any] = {
            "dtype_family": family,
            "required": True,
            "missing_rate_ceiling": min(
                1.0, round(missing_rate + max(0.05, missing_rate * 0.25), 4)
            ),
        }
        unique = series.dropna().unique()
        if family in {"categorical", "boolean"} and len(unique) <= 50:
            rule["allowed_values"] = sorted(str(value) for value in unique)
        columns[name] = rule
    return {
        "contract_version": 1,
        "target": target,
        "columns": columns,
    }


def validate_contract(
    df: pd.DataFrame, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return contract violations without changing the table."""
    violations: list[dict[str, Any]] = []
    expected = contract["columns"]
    for column, rule in expected.items():
        if column not in df.columns:
            violations.append(
                {
                    "severity": "high",
                    "column": column,
                    "check": "missing_column",
                    "message": "Required column is missing.",
                }
            )
            continue
        series = df[column]
        observed_family = _dtype_family(series)
        if observed_family != rule["dtype_family"]:
            violations.append(
                {
                    "severity": "high",
                    "column": column,
                    "check": "dtype_family",
                    "message": (
                        f"Expected {rule['dtype_family']}; found "
                        f"{observed_family}."
                    ),
                }
            )
        missing_rate = float(series.isna().mean())
        if missing_rate > rule["missing_rate_ceiling"]:
            violations.append(
                {
                    "severity": "medium",
                    "column": column,
                    "check": "missing_rate",
                    "message": (
                        f"Missing rate {missing_rate:.1%} exceeds the "
                        f"{rule['missing_rate_ceiling']:.1%} ceiling."
                    ),
                }
            )
        allowed = set(rule.get("allowed_values", []))
        if allowed:
            observed = set(series.dropna().astype(str).unique())
            unknown = sorted(observed - allowed)
            if unknown:
                violations.append(
                    {
                        "severity": "medium",
                        "column": column,
                        "check": "unknown_values",
                        "message": (
                            f"New values: {', '.join(unknown[:10])}"
                        ),
                    }
                )
    for column in sorted(set(df.columns) - set(expected)):
        violations.append(
            {
                "severity": "low",
                "column": column,
                "check": "extra_column",
                "message": "Column was not present in the reference data.",
            }
        )
    return violations


def contract_json(contract: dict[str, Any]) -> bytes:
    return json.dumps(contract, indent=2).encode("utf-8")


def _numeric_psi(reference: pd.Series, current: pd.Series) -> float | None:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < 20 or len(current) < 20:
        return None
    boundaries = np.unique(
        reference.quantile(np.linspace(0, 1, 11)).to_numpy()
    )
    if len(boundaries) < 3:
        return 0.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=boundaries)[0]
    cur_counts = np.histogram(current, bins=boundaries)[0]
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(
        np.sum((cur_share - ref_share) * np.log(cur_share / ref_share))
    )


def compare_datasets(
    reference: pd.DataFrame, current: pd.DataFrame
) -> list[dict[str, Any]]:
    """Compare schema, missingness, and distributions to a reference."""
    results: list[dict[str, Any]] = []
    all_columns = sorted(set(reference.columns) | set(current.columns))
    for column in all_columns:
        if column not in reference.columns:
            results.append(
                {
                    "column": column,
                    "check": "schema",
                    "severity": "low",
                    "detail": "New column.",
                }
            )
            continue
        if column not in current.columns:
            results.append(
                {
                    "column": column,
                    "check": "schema",
                    "severity": "high",
                    "detail": "Reference column is missing.",
                }
            )
            continue
        missing_change = float(
            current[column].isna().mean()
            - reference[column].isna().mean()
        )
        if abs(missing_change) >= 0.05:
            results.append(
                {
                    "column": column,
                    "check": "missingness_shift",
                    "severity": "medium",
                    "detail": f"Missingness changed by {missing_change:+.1%}.",
                }
            )
        if (
            _dtype_family(reference[column]) == "numeric"
            and _dtype_family(current[column]) == "numeric"
        ):
            psi = _numeric_psi(reference[column], current[column])
            if psi is not None and psi >= 0.20:
                results.append(
                    {
                        "column": column,
                        "check": "numeric_drift",
                        "severity": "medium" if psi < 0.50 else "high",
                        "detail": f"Population Stability Index: {psi:.3f}.",
                    }
                )
        else:
            ref_share = reference[column].astype("string").value_counts(
                normalize=True, dropna=False
            )
            cur_share = current[column].astype("string").value_counts(
                normalize=True, dropna=False
            )
            values = ref_share.index.union(cur_share.index)
            distance = float(
                0.5
                * (
                    ref_share.reindex(values, fill_value=0)
                    - cur_share.reindex(values, fill_value=0)
                ).abs().sum()
            )
            if distance >= 0.20:
                results.append(
                    {
                        "column": column,
                        "check": "category_drift",
                        "severity": "medium" if distance < 0.40 else "high",
                        "detail": (
                            f"Category distribution distance: {distance:.3f}."
                        ),
                    }
                )
    return results
