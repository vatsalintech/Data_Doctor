"""Target-agnostic, rule-based checks for classification tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ID_UNIQUE_THRESHOLD = 0.95
ID_MIN_UNIQUE_VALUES = 20
RARE_CLASS_THRESHOLD = 0.01
LEAKAGE_MATCH_THRESHOLD = 0.95
MOSTLY_MISSING_THRESHOLD = 0.80
MISSING_LABEL_TOKENS = {
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


@dataclass(frozen=True)
class Finding:
    issue_type: str
    scope: str
    severity: str
    found: bool
    evidence: dict[str, Any]
    suggested_action: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where blank strings and '?' count as missing."""
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


def dataset_vitals(df: pd.DataFrame) -> dict[str, Any]:
    cells = int(df.shape[0] * df.shape[1])
    missing = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing,
        "missing_percent": (missing / cells * 100) if cells else 0.0,
        "exact_duplicate_extras": int(df.duplicated().sum()),
    }


def is_likely_id(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return False
    unique_count = int(non_missing.nunique())
    return (
        unique_count >= ID_MIN_UNIQUE_VALUES
        and unique_count / len(non_missing) >= ID_UNIQUE_THRESHOLD
    )


def id_like_columns(df: pd.DataFrame) -> list[str]:
    return [
        column for column in df.columns if is_likely_id(df[column])
    ]


def validate_classification_table(
    df: pd.DataFrame,
    target: str,
    *,
    minimum_rows: int = 10,
) -> list[str]:
    errors: list[str] = []
    if target not in df.columns:
        return [f"Target column {target!r} was not found."]
    if len(df) < minimum_rows:
        errors.append(
            f"At least {minimum_rows} rows are needed; found {len(df)}."
        )
    if len(df.columns) < 2:
        errors.append("At least one feature column is required.")

    target_values = df[target].dropna()
    class_count = int(target_values.nunique())
    if class_count < 2:
        errors.append("The target must contain at least two classes.")
    if is_likely_id(target_values):
        errors.append(
            "The selected target looks like an ID because almost every "
            "value is unique."
        )
    if target_values.empty:
        errors.append("The target has no usable values.")
    return errors


def missing_findings(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    row_count = len(df)
    for column in df.columns:
        count = int(df[column].isna().sum())
        percent = (count / row_count * 100) if row_count else 0.0
        found = count > 0
        severity = (
            "none"
            if not found
            else "low"
            if percent < 5
            else "medium"
        )
        mostly_missing = percent / 100 >= MOSTLY_MISSING_THRESHOLD
        if not found:
            action = None
            reason = "No issue found."
        elif mostly_missing:
            action = "drop_column"
            reason = "Most values are missing, so the column may add little."
        elif pd.api.types.is_numeric_dtype(df[column]):
            action = "impute_median"
            reason = "Median filling is resistant to extreme numeric values."
        else:
            action = "impute_mode"
            reason = "Mode filling uses the most common category."
        findings.append(
            Finding(
                issue_type="missing_values",
                scope=column,
                severity=severity,
                found=found,
                evidence={"count": count, "percent": percent},
                suggested_action=action,
                reason=reason,
            )
        )
    return findings


def duplicate_findings(df: pd.DataFrame, target: str) -> list[Finding]:
    exact_count = int(df.duplicated().sum())
    features = [column for column in df.columns if column != target]

    conflict_groups = 0
    conflict_rows = 0
    if features:
        possible = df.duplicated(subset=features, keep=False)
        grouped = (
            df.loc[possible]
            .groupby(features, dropna=False)[target]
            .nunique(dropna=True)
        )
        conflict_groups = int((grouped > 1).sum())
        conflict_keys = grouped[grouped > 1].index
        if conflict_groups:
            group_sizes = (
                df.loc[possible]
                .groupby(features, dropna=False)
                .size()
            )
            conflict_rows = int(group_sizes.loc[conflict_keys].sum())

    return [
        Finding(
            issue_type="exact_duplicates",
            scope="dataset",
            severity="high" if exact_count else "none",
            found=exact_count > 0,
            evidence={"extra_rows": exact_count},
            suggested_action="drop_exact_duplicates"
            if exact_count
            else None,
            reason=(
                "Copies can land in both train and test."
                if exact_count
                else "No issue found."
            ),
        ),
        Finding(
            issue_type="conflicting_labels",
            scope="dataset",
            severity="high" if conflict_groups else "none",
            found=conflict_groups > 0,
            evidence={
                "groups": conflict_groups,
                "rows": conflict_rows,
            },
            suggested_action="drop_conflicting_rows"
            if conflict_groups
            else None,
            reason=(
                "Identical features should not silently have different labels."
                if conflict_groups
                else "No issue found."
            ),
        ),
    ]


def outlier_findings(df: pd.DataFrame, target: str) -> list[Finding]:
    findings: list[Finding] = []
    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if column == target:
            continue
        values = df[column].dropna()
        if values.nunique() <= 2:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[column] < low) | (df[column] > high)
        count = int(mask.sum())
        percent = (count / len(df) * 100) if len(df) else 0.0
        findings.append(
            Finding(
                issue_type="outliers",
                scope=column,
                severity=(
                    "none"
                    if count == 0
                    else "low"
                    if percent < 1
                    else "medium"
                ),
                found=count > 0,
                evidence={
                    "count": count,
                    "percent": percent,
                    "lower_bound": float(low),
                    "upper_bound": float(high),
                },
                suggested_action="cap_outliers" if count else None,
                reason=(
                    "IQR found values far outside the middle of the data."
                    if count
                    else "No issue found."
                ),
            )
        )
    return findings


def imbalance_finding(df: pd.DataFrame, target: str) -> Finding:
    counts = df[target].value_counts(dropna=True)
    ratio = (
        float(counts.max() / counts.min())
        if len(counts) >= 2 and counts.min() > 0
        else float("inf")
    )
    found = ratio >= 2.0
    return Finding(
        issue_type="class_imbalance",
        scope="dataset",
        severity="medium" if found else "none",
        found=found,
        evidence={
            "counts": {str(k): int(v) for k, v in counts.items()},
            "ratio": ratio,
        },
        suggested_action="class_weight_balanced" if found else None,
        reason=(
            "The largest class is at least twice the smallest."
            if found
            else "No issue found."
        ),
    )


def suspicious_label_findings(
    df: pd.DataFrame, target: str
) -> list[Finding]:
    values = df[target]
    normalized = values.astype("string").str.strip().str.lower()
    weird_mask = normalized.isin(MISSING_LABEL_TOKENS) | values.isna()
    weird_count = int(weird_mask.sum())

    shares = values.value_counts(normalize=True, dropna=True)
    rare = shares[shares < RARE_CLASS_THRESHOLD]
    rare_rows = int(values.isin(rare.index).sum())

    return [
        Finding(
            issue_type="weird_labels",
            scope=target,
            severity="high" if weird_count else "none",
            found=weird_count > 0,
            evidence={"rows": weird_count},
            suggested_action="review_or_drop_label_rows"
            if weird_count
            else None,
            reason=(
                "Some target values look like missing labels."
                if weird_count
                else "No issue found."
            ),
        ),
        Finding(
            issue_type="rare_classes",
            scope=target,
            severity="low" if len(rare) else "none",
            found=len(rare) > 0,
            evidence={
                "classes": {
                    str(label): float(share)
                    for label, share in rare.items()
                },
                "rows": rare_rows,
            },
            suggested_action="review_rare_classes"
            if len(rare)
            else None,
            reason=(
                "A class represents less than 1% of usable labels."
                if len(rare)
                else "No issue found."
            ),
        ),
    ]


def leakage_findings(df: pd.DataFrame, target: str) -> list[Finding]:
    findings: list[Finding] = []
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        if is_likely_id(series):
            unique_rate = float(
                series.dropna().nunique() / len(series.dropna())
            )
            findings.append(
                Finding(
                    issue_type="id_like",
                    scope=column,
                    severity="low",
                    found=True,
                    evidence={"unique_rate": unique_rate},
                    suggested_action="drop_column",
                    reason="Almost every value is unique.",
                )
            )

        pair = df[[column, target]].dropna()
        if pair.empty:
            continue
        matches = (
            pair[column].astype(str) == pair[target].astype(str)
        )
        match_rate = float(matches.mean())
        reverse_rate = 0.0
        if pair[column].nunique() == pair[target].nunique() == 2:
            column_codes, _ = pd.factorize(pair[column])
            target_codes, _ = pd.factorize(pair[target])
            reverse_rate = float((column_codes != target_codes).mean())
        if max(match_rate, reverse_rate) >= LEAKAGE_MATCH_THRESHOLD:
            findings.append(
                Finding(
                    issue_type="target_copy",
                    scope=column,
                    severity="high",
                    found=True,
                    evidence={
                        "match_rate": max(match_rate, reverse_rate)
                    },
                    suggested_action="drop_column",
                    reason=(
                        "This feature almost directly reveals the target."
                    ),
                )
            )
    return findings


def diagnose(df: pd.DataFrame, target: str) -> list[Finding]:
    prepared = prepare_data(df)
    errors = validate_classification_table(prepared, target)
    if errors:
        raise ValueError(" ".join(errors))
    return [
        *missing_findings(prepared),
        *duplicate_findings(prepared, target),
        imbalance_finding(prepared, target),
        *outlier_findings(prepared, target),
        *suspicious_label_findings(prepared, target),
        *leakage_findings(prepared, target),
    ]


def finding_rows(findings: list[Finding]) -> pd.DataFrame:
    """Flatten findings for a compact table in scripts and Streamlit."""
    rows = []
    for finding in findings:
        row = {
            "issue": finding.issue_type,
            "scope": finding.scope,
            "severity": finding.severity,
            "status": "Issue found" if finding.found else "No issue found",
            "suggested_action": finding.suggested_action or "",
            "reason": finding.reason or "",
        }
        row.update(
            {
                f"evidence_{key}": value
                for key, value in finding.evidence.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).replace({np.nan: None})

