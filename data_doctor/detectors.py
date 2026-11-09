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
SUSPICIOUS_PREDICTION_THRESHOLD = 0.98
SUSPICIOUS_PREDICTION_UPLIFT = 0.20
SUSPICIOUS_PREDICTION_MIN_COVERAGE = 0.80
SUSPICIOUS_PREDICTION_MIN_GROUP_SIZE = 5
SUSPICIOUS_PREDICTION_MAX_UNIQUE_RATE = 0.50
MOSTLY_MISSING_THRESHOLD = 0.80
STORED_TEXT_MIN_VALUES = 3
# Closed list of spellings that mean "no value" across real CSV exports.
# This is not a guess about category names such as female/f/0.
TEXT_MISSING_TOKENS = {
    "",
    "?",
    "??",
    "???",
    "*",
    "-",
    "--",
    ".",
    "na",
    "n/a",
    "n.a",
    "n.a.",
    "na.",
    "#n/a",
    "#na",
    "#n/a n/a",
    "<na>",
    "none",
    "null",
    "nil",
    "nan",
    "nat",
    "unknown",
    "missing",
    "not available",
    "not applicable",
    "unavailable",
    "unspecified",
    "not known",
    "not specified",
    "#value!",
    "#div/0!",
    "#ref!",
    "#name?",
    "#null!",
    "#num!",
}
# Extreme sentinels that are almost never real measurements. -1 and 999 are
# left in place and reported, because they can be legitimate values.
NUMERIC_SENTINEL_VALUES = (-99, -999, -9999)
PLACEHOLDER_REVIEW_VALUES = (-1, 999, 9999)
MOSTLY_NUMERIC_THRESHOLD = 0.80
PLACEHOLDER_MAX_SHARE = 0.05
TARGET_LABEL_REVIEW_MIN = 10
FEATURE_STRAY_UNIQUE_MIN = 15
FEATURE_STRAY_RARE_MIN = 5
MISSING_LABEL_TOKENS = TEXT_MISSING_TOKENS


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


def _is_text_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _text_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def missing_token_mask(series: pd.Series) -> pd.Series:
    """True where a cell is empty or uses a known stand-in for missing."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.isna()
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.isna() | series.isin(NUMERIC_SENTINEL_VALUES)
    if not _is_text_like(series):
        return series.isna()

    key = _text_key(series)
    text_missing = key.isin(TEXT_MISSING_TOKENS)
    parsed = pd.to_numeric(key, errors="coerce")
    candidate = series.notna() & ~text_missing
    mostly_numeric = False
    if int(candidate.sum()) >= 3:
        mostly_numeric = bool(
            parsed.loc[candidate].notna().mean() >= MOSTLY_NUMERIC_THRESHOLD
        )
    sentinel = (
        parsed.isin(NUMERIC_SENTINEL_VALUES)
        if mostly_numeric
        else False
    )
    return series.isna() | text_missing | sentinel


def normalize_missing_tokens(series: pd.Series) -> pd.Series:
    """Return a copy where known missing stand-ins are real missing values.

    Diagnosis, the missing count on a card, and fill or drop actions all
    use this rule. Otherwise a card can say the missing count is zero while
    the problem list still sees `?`, `unknown`, or `-999`.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("object")
    mask = missing_token_mask(series)
    if not bool(mask.any()):
        return series
    return series.mask(mask, pd.NA)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where known missing stand-ins count as missing."""
    prepared = df.copy()
    text_columns = prepared.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in text_columns:
        prepared[column] = normalize_missing_tokens(prepared[column])
    numeric_columns = prepared.select_dtypes(include="number").columns
    for column in numeric_columns:
        prepared[column] = normalize_missing_tokens(prepared[column])
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
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return False
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
    elif class_count > max(20, int(len(target_values) * 0.20)):
        errors.append(
            "The target has too many classes for a useful v1 "
            "classification model."
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


def stored_as_text_findings(
    df: pd.DataFrame,
    target: str,
) -> list[Finding]:
    """Find feature columns whose non-missing text has one safer type."""
    findings: list[Finding] = []
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns
    for column in text_columns:
        if column == target:
            continue
        values = df[column].dropna().astype("string").str.strip()
        if (
            len(values) < STORED_TEXT_MIN_VALUES
            or values.nunique() <= 2
        ):
            continue

        numeric = pd.to_numeric(values, errors="coerce")
        has_leading_zero_identifiers = values.str.match(
            r"^[+-]?0\d+$"
        ).any()
        all_finite = bool(
            numeric.notna().all()
            and np.isfinite(numeric.to_numpy(dtype=float)).all()
        )
        if all_finite and not has_leading_zero_identifiers:
            findings.append(
                Finding(
                    issue_type="stored_as_text",
                    scope=column,
                    severity="medium",
                    found=True,
                    evidence={
                        "inferred_type": "number",
                        "parseable_values": int(len(values)),
                        "parseable_percent": 100.0,
                    },
                    suggested_action="convert_to_number",
                    reason=(
                        "Every non-missing value looks numeric, but the "
                        "column is stored as text."
                    ),
                )
            )
            continue

        looks_date_shaped = values.str.contains(
            r"[-/:T]|[A-Za-z]{3,}",
            regex=True,
        )
        if not looks_date_shaped.mean() >= 0.80:
            continue
        parsed_dates = pd.to_datetime(
            values,
            errors="coerce",
            format="mixed",
        )
        if parsed_dates.notna().all():
            findings.append(
                Finding(
                    issue_type="stored_as_text",
                    scope=column,
                    severity="medium",
                    found=True,
                    evidence={
                        "inferred_type": "date",
                        "parseable_values": int(len(values)),
                        "parseable_percent": 100.0,
                    },
                    suggested_action="convert_to_datetime",
                    reason=(
                        "Every non-missing value looks like a date, but "
                        "the column is stored as text."
                    ),
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
    # Nulls are already reported as missing values. Weird labels are only
    # non-empty stand-ins that normalization did not convert.
    weird_mask = normalized.isin(MISSING_LABEL_TOKENS) & values.notna()
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
    usable_target_rows = int(df[target].notna().sum())
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        id_like = is_likely_id(series)
        if id_like:
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
        direct_match_rate = max(match_rate, reverse_rate)
        if direct_match_rate >= LEAKAGE_MATCH_THRESHOLD:
            findings.append(
                Finding(
                    issue_type="target_copy",
                    scope=column,
                    severity="high",
                    found=True,
                    evidence={
                        "match_rate": direct_match_rate
                    },
                    suggested_action="drop_column",
                    reason=(
                        "This feature almost directly reveals the target."
                    ),
                )
            )
            continue

        coverage = (
            len(pair) / usable_target_rows if usable_target_rows else 0.0
        )
        unique_rate = pair[column].nunique() / len(pair)
        if (
            id_like
            or coverage < SUSPICIOUS_PREDICTION_MIN_COVERAGE
            or unique_rate > SUSPICIOUS_PREDICTION_MAX_UNIQUE_RATE
        ):
            continue
        group_sizes = pair.groupby(column, dropna=False).size()
        supported_values = group_sizes[
            group_sizes >= SUSPICIOUS_PREDICTION_MIN_GROUP_SIZE
        ].index
        supported = pair[pair[column].isin(supported_values)]
        if len(supported) / len(pair) < SUSPICIOUS_PREDICTION_MIN_COVERAGE:
            continue
        grouped = supported.groupby(
            column, dropna=False
        )[target].value_counts()
        predictable_rows = int(grouped.groupby(level=0).max().sum())
        prediction_rate = predictable_rows / len(supported)
        baseline_rate = float(
            supported[target].value_counts().max() / len(supported)
        )
        if (
            prediction_rate >= SUSPICIOUS_PREDICTION_THRESHOLD
            and prediction_rate - baseline_rate
            >= SUSPICIOUS_PREDICTION_UPLIFT
        ):
            findings.append(
                Finding(
                    issue_type="suspicious_predictor",
                    scope=column,
                    severity="high",
                    found=True,
                    evidence={
                        "prediction_rate": prediction_rate,
                        "baseline_rate": baseline_rate,
                        "coverage": coverage,
                    },
                    suggested_action="drop_column",
                    reason=(
                        "This feature almost perfectly maps to the target, "
                        "which may indicate leakage."
                    ),
                )
            )
    return findings


def _variant_groups(values: pd.Series) -> list[list[str]]:
    """Labels that match after trimming and ignoring letter case."""
    text = values.astype("string")
    folded = text.str.strip().str.lower()
    groups: list[list[str]] = []
    for _, subset in text.groupby(folded, sort=False):
        forms = list(dict.fromkeys(str(item) for item in subset.tolist()))
        if len(forms) > 1:
            groups.append(forms[:4])
    return groups


def _placeholder_counts(values: pd.Series) -> list[str]:
    parsed = pd.to_numeric(
        values.astype("string").str.strip(), errors="coerce"
    )
    descriptions: list[str] = []
    row_count = max(len(values), 1)
    for token in PLACEHOLDER_REVIEW_VALUES:
        count = int((parsed == token).sum())
        if count == 0 or count / row_count >= PLACEHOLDER_MAX_SHARE:
            continue
        if token < 0:
            other_negatives = int(((parsed < 0) & (parsed != token)).sum())
            if other_negatives:
                continue
        descriptions.append(f"{token:g} ({count})")
    return descriptions


def _mostly_numeric_text(values: pd.Series) -> bool:
    if len(values) < 3:
        return False
    parsed = pd.to_numeric(
        values.astype("string").str.strip(), errors="coerce"
    )
    return bool(parsed.notna().mean() >= MOSTLY_NUMERIC_THRESHOLD)


def label_review_findings(
    df: pd.DataFrame, target: str
) -> list[Finding]:
    """Flag label mess that must be reviewed, without rewriting it.

    Case, spaces, and alternate spellings are not merged. A wrong merge
    changes what the column means.
    """
    findings: list[Finding] = []
    text_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in text_columns:
        series = df[column]
        if is_likely_id(series):
            continue
        values = series.dropna()
        if values.empty:
            continue
        reasons: list[str] = []
        numeric_text = _mostly_numeric_text(values)
        if numeric_text:
            placeholders = _placeholder_counts(values)
            if placeholders:
                reasons.append(
                    "Possible placeholder numbers were left as real "
                    "measurements: "
                    + ", ".join(placeholders)
                    + "."
                )
            parsed = pd.to_numeric(
                values.astype("string").str.strip(), errors="coerce"
            )
            unparsed = values.astype("string").str.strip()[parsed.isna()]
            if len(unparsed):
                examples = list(
                    dict.fromkeys(str(item) for item in unparsed.tolist())
                )[:4]
                label_count = int(unparsed.nunique())
                noun = "label" if label_count == 1 else "labels"
                reasons.append(
                    f"{label_count} {noun} could not be read as numbers ("
                    + ", ".join(examples)
                    + ")."
                )
        else:
            groups = _variant_groups(values)
            if groups:
                shown = [", ".join(group) for group in groups[:3]]
                extra = len(groups) - len(shown)
                detail = "; ".join(shown)
                if extra > 0:
                    detail = f"{detail}; and {extra} more group(s)"
                reasons.append(
                    "Some labels differ only by letter case or spaces "
                    f"({detail})."
                )
            visible = values.astype("string").str.strip()
            folded = visible.str.lower()
            label_count = int(visible.nunique())
            rare_labels = int(
                (
                    folded.value_counts(normalize=True) < RARE_CLASS_THRESHOLD
                ).sum()
            )
            if (
                column == target
                and label_count >= TARGET_LABEL_REVIEW_MIN
            ):
                reasons.append(
                    f"The target has {label_count} distinct labels. "
                    "Some of them may be the same class written in "
                    "different ways."
                )
            elif (
                column != target
                and label_count >= FEATURE_STRAY_UNIQUE_MIN
                and rare_labels >= FEATURE_STRAY_RARE_MIN
            ):
                reasons.append(
                    f"This column has {label_count} distinct labels, "
                    f"and {rare_labels} of them are uncommon."
                )
        if not reasons:
            continue
        severity = "low"
        if column == target:
            severity = "high"
        elif numeric_text:
            severity = "medium"
        findings.append(
            Finding(
                issue_type="manual_review",
                scope=column,
                severity=severity,
                found=True,
                evidence={
                    "labels": int(
                        values.astype("string").str.strip().str.lower().nunique()
                    )
                },
                suggested_action=None,
                reason=(
                    " ".join(reasons)
                    + " Data Doctor will not guess a replacement or merge "
                    "these values. Review them and fix the file yourself."
                ),
            )
        )

    spaced_names = [
        str(column)
        for column in df.columns
        if str(column) != str(column).strip() or str(column).strip() == ""
    ]
    if spaced_names:
        shown_names = ", ".join(repr(name) for name in spaced_names[:6])
        findings.append(
            Finding(
                issue_type="manual_review",
                scope="dataset",
                severity="low",
                found=True,
                evidence={"columns": shown_names},
                suggested_action=None,
                reason=(
                    "These column names are blank or have extra spaces: "
                    f"{shown_names}. Rename them in the file. Data Doctor "
                    "will not rename columns."
                ),
            )
        )

    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if column == target:
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        descriptions: list[str] = []
        row_count = max(len(values), 1)
        for token in PLACEHOLDER_REVIEW_VALUES:
            count = int((values == token).sum())
            if count == 0 or count / row_count >= PLACEHOLDER_MAX_SHARE:
                continue
            if token < 0 and int(((values < 0) & (values != token)).sum()):
                continue
            descriptions.append(f"{token:g} ({count})")
        if not descriptions:
            continue
        findings.append(
            Finding(
                issue_type="manual_review",
                scope=column,
                severity="medium",
                found=True,
                evidence={"placeholders": ", ".join(descriptions)},
                suggested_action=None,
                reason=(
                    "Possible placeholder numbers were left as real "
                    "measurements: "
                    + ", ".join(descriptions)
                    + ". Data Doctor will not treat them as missing, "
                    "because they can be legitimate values. Review them "
                    "yourself."
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
        *stored_as_text_findings(prepared, target),
        *duplicate_findings(prepared, target),
        imbalance_finding(prepared, target),
        *outlier_findings(prepared, target),
        *suspicious_label_findings(prepared, target),
        *leakage_findings(prepared, target),
        *label_review_findings(prepared, target),
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

