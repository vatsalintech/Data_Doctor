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


def _missingness_slack(missing_rate: float) -> float:
    """Allowed movement before missingness is unexpected.

    The cushion is at least 5 percentage points and at most 10, so a
    mostly-missing column can still fail when it becomes fully missing.
    """
    rate = float(missing_rate)
    return max(0.05, min(0.10, rate * 0.25))


def _missingness_ceiling(missing_rate: float) -> float:
    rate = float(missing_rate)
    if rate >= 1:
        return 1.0
    return round(min(rate + _missingness_slack(rate), 0.9999), 4)


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
            "missing_rate_ceiling": _missingness_ceiling(missing_rate),
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


def _stability_index(
    reference_counts: np.ndarray,
    current_counts: np.ndarray,
) -> float:
    """PSI with a tiny share substituted for an empty bin.

    Bins are not renormalized after that substitution. An empty bin stays
    visible, and the logarithm stays finite.
    """
    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()
    reference_share = np.where(
        reference_counts == 0, 1e-6, reference_share
    )
    current_share = np.where(current_counts == 0, 1e-6, current_share)
    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )


def _discrete_psi(reference: pd.Series, current: pd.Series) -> float:
    """Compare numeric values when quantile bins collapse to one or two edges."""
    categories = list(pd.unique(reference))
    reference_counts = (
        reference.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )
    current_counts = (
        current.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )
    unseen = float((~current.isin(categories)).sum())
    reference_counts = np.append(reference_counts, 0.0)
    current_counts = np.append(current_counts, unseen)
    return _stability_index(reference_counts, current_counts)


def _numeric_psi(reference: pd.Series, current: pd.Series) -> float | None:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < 20 or len(current) < 20:
        return None
    edges = np.unique(
        reference.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float)
    )
    if len(edges) < 3:
        return _discrete_psi(reference, current)
    edges = edges.astype(float, copy=True)
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(
        reference.to_numpy(dtype=float), bins=edges
    )[0]
    current_counts = np.histogram(
        current.to_numpy(dtype=float), bins=edges
    )[0]
    return _stability_index(
        reference_counts.astype(float),
        current_counts.astype(float),
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
        reference_missing = float(reference[column].isna().mean())
        missing_change = float(
            current[column].isna().mean() - reference_missing
        )
        if abs(missing_change) > _missingness_slack(reference_missing):
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


def classify_comparison(
    contract_issues: list[dict[str, Any]],
    drift: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep schema and expected missingness out of the distribution section.

    Schema rows stay in the contract findings. Missingness above the ceiling
    stays there too. A missingness drop is not a contract breach, so it is
    reported on its own. PSI and category shifts are the distribution signals.
    """
    contract_missing = {
        row["column"]
        for row in contract_issues
        if row["check"] == "missing_rate"
    }
    return {
        "distribution": [
            row
            for row in drift
            if row["check"] in {"numeric_drift", "category_drift"}
        ],
        "unexpected_missingness": [
            row
            for row in drift
            if row["check"] == "missingness_shift"
            and row["column"] not in contract_missing
        ],
    }
