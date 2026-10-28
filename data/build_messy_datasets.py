"""Build reproducible messy demo datasets without changing the originals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "Original"
MESSY_DIR = ROOT / "Messy"
MANIFEST_PATH = MESSY_DIR / "issue_manifest.json"
SEED = 42


def _stable_source_ids(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Give identical source rows the same ID while keeping IDs mostly unique."""
    hashes = pd.util.hash_pandas_object(df, index=False)
    codes, _ = pd.factorize(hashes, sort=False)
    return pd.Series(
        [f"{prefix}-{code + 1:07d}" for code in codes],
        index=df.index,
        dtype="string",
    )


def _choose(
    rng: np.random.Generator,
    index: pd.Index,
    count: int,
    used: set[int],
) -> np.ndarray:
    pool = np.array([int(i) for i in index if int(i) not in used])
    chosen = rng.choice(pool, size=count, replace=False)
    used.update(int(i) for i in chosen)
    return chosen


def _set_missing(
    df: pd.DataFrame,
    rng: np.random.Generator,
    column: str,
    fraction: float,
) -> int:
    count = int(round(len(df) * fraction))
    chosen = rng.choice(df.index.to_numpy(), size=count, replace=False)
    df.loc[chosen, column] = np.nan
    return count


def _add_mostly_empty_column(
    df: pd.DataFrame,
    rng: np.random.Generator,
    column: str,
    values: list[str],
) -> None:
    filled_count = int(round(len(df) * 0.10))
    filled = rng.choice(
        df.index.to_numpy(), size=filled_count, replace=False
    )
    df[column] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[filled, column] = rng.choice(values, size=filled_count)


def _flip_binary(
    values: pd.Series,
    mapping: dict[Any, Any],
) -> pd.Series:
    flipped = values.map(mapping)
    if flipped.isna().any():
        raise ValueError("Target contains a value outside the flip mapping.")
    return flipped


def _finish_dataset(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target: str,
    target_copy: str,
    flip_mapping: dict[Any, Any],
    exact_count: int,
    conflict_count: int,
    noisy_label_count: int,
    used: set[int],
) -> pd.DataFrame:
    """Add leakage, noisy labels, exact copies, and label conflicts."""
    df[target_copy] = df[target]

    noisy_indices = _choose(
        rng, df.index, noisy_label_count, used
    )
    df.loc[noisy_indices, target] = _flip_binary(
        df.loc[noisy_indices, target], flip_mapping
    )

    exact_indices = _choose(rng, df.index, exact_count, used)
    conflict_indices = _choose(
        rng, df.index, conflict_count, used
    )

    exact_rows = df.loc[exact_indices].copy()
    conflicting_rows = df.loc[conflict_indices].copy()
    conflicting_rows[target] = _flip_binary(
        conflicting_rows[target], flip_mapping
    )

    return pd.concat(
        [df, exact_rows, conflicting_rows], ignore_index=True
    )


def _observed_evidence(
    df: pd.DataFrame,
    target: str,
    target_copy: str,
    id_column: str,
) -> dict[str, Any]:
    feature_columns = [
        column
        for column in df.columns
        if column not in {target, target_copy}
    ]
    possible_conflicts = df.duplicated(
        subset=feature_columns, keep=False
    )
    conflicts = (
        df.loc[possible_conflicts]
        .groupby(feature_columns, dropna=False)[target]
        .nunique()
    )

    outliers: dict[str, int] = {}
    for column in df.select_dtypes(include="number").columns:
        if column == target:
            continue
        values = df[column].dropna()
        if values.nunique() <= 2:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < low) | (values > high)).sum())
        if count:
            outliers[column] = count

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count
        },
        "exact_duplicate_extras": int(df.duplicated().sum()),
        "conflicting_label_groups": int((conflicts > 1).sum()),
        "target_counts": {
            str(label): int(count)
            for label, count in df[target].value_counts().items()
        },
        "target_copy_match_count": int(
            (df[target_copy] == df[target]).sum()
        ),
        "target_copy_match_rate": float(
            (df[target_copy] == df[target]).mean()
        ),
        "id_unique_rate": float(df[id_column].nunique() / len(df)),
        "iqr_outlier_counts": outliers,
    }


def build_telco() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(
        ORIGINAL_DIR / "Telco-Customer-Churn.csv"
    )
    df = source.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    rng = np.random.default_rng(SEED)

    planted_missing = {
        "tenure": _set_missing(df, rng, "tenure", 0.05),
        "MonthlyCharges": _set_missing(
            df, rng, "MonthlyCharges", 0.12
        ),
        "TotalCharges": _set_missing(
            df, rng, "TotalCharges", 0.08
        ),
        "Contract": _set_missing(df, rng, "Contract", 0.10),
        "PaymentMethod": _set_missing(
            df, rng, "PaymentMethod", 0.06
        ),
    }
    _add_mostly_empty_column(
        df, rng, "promo_code", ["SAVE10", "SAVE20", "WELCOME"]
    )

    used: set[int] = set()
    outlier_plan = {
        "tenure": (25, 999),
        "MonthlyCharges": (25, 9999),
        "TotalCharges": (20, 999999),
    }
    for column, (count, value) in outlier_plan.items():
        chosen = _choose(rng, df.index, count, used)
        df.loc[chosen, column] = value

    messy = _finish_dataset(
        df=df,
        rng=rng,
        target="Churn",
        target_copy="ChurnCopy",
        flip_mapping={"Yes": "No", "No": "Yes"},
        exact_count=60,
        conflict_count=35,
        noisy_label_count=40,
        used=used,
    )
    metadata = {
        "source_file": "Original/Telco-Customer-Churn.csv",
        "output_file": "Messy/telco_messy.csv",
        "target": "Churn",
        "id_column": "customerID",
        "target_copy": "ChurnCopy",
        "source_rows": int(len(source)),
        "source_exact_duplicate_extras": int(
            source.duplicated().sum()
        ),
        "planted": {
            "missing_requested": planted_missing,
            "mostly_empty_column": "promo_code",
            "outliers": {
                column: {"rows": count, "value": value}
                for column, (count, value) in outlier_plan.items()
            },
            "exact_copies": 60,
            "conflicting_label_copies": 35,
            "noisy_label_flips": 40,
        },
        "observed": _observed_evidence(
            messy, "Churn", "ChurnCopy", "customerID"
        ),
    }
    return messy, metadata


def build_adult() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(ORIGINAL_DIR / "adult.csv")
    df = source.copy()
    source_question_marks = {
        column: int(
            df[column].astype("string").str.strip().eq("?").sum()
        )
        for column in df.select_dtypes(include="object").columns
    }
    source_question_marks = {
        column: count
        for column, count in source_question_marks.items()
        if count
    }
    df.replace(r"^\s*\?\s*$", pd.NA, regex=True, inplace=True)
    df.insert(0, "record_id", _stable_source_ids(source, "ADULT"))
    rng = np.random.default_rng(SEED + 1)

    planted_missing = {
        "age": _set_missing(df, rng, "age", 0.04),
        "hours.per.week": _set_missing(
            df, rng, "hours.per.week", 0.07
        ),
        "education": _set_missing(df, rng, "education", 0.06),
        "marital.status": _set_missing(
            df, rng, "marital.status", 0.08
        ),
    }
    _add_mostly_empty_column(
        df, rng, "review_code", ["REVIEW_A", "REVIEW_B"]
    )

    used: set[int] = set()
    outlier_plan = {
        "age": (80, 999),
        "hours.per.week": (80, 999),
        "capital.gain": (60, 9999999),
    }
    for column, (count, value) in outlier_plan.items():
        chosen = _choose(rng, df.index, count, used)
        df.loc[chosen, column] = value

    messy = _finish_dataset(
        df=df,
        rng=rng,
        target="income",
        target_copy="income_copy",
        flip_mapping={"<=50K": ">50K", ">50K": "<=50K"},
        exact_count=100,
        conflict_count=50,
        noisy_label_count=50,
        used=used,
    )
    metadata = {
        "source_file": "Original/adult.csv",
        "output_file": "Messy/adult_messy.csv",
        "target": "income",
        "id_column": "record_id",
        "target_copy": "income_copy",
        "source_rows": int(len(source)),
        "source_exact_duplicate_extras": int(
            source.duplicated().sum()
        ),
        "source_question_mark_cells": source_question_marks,
        "planted": {
            "question_marks_converted_to_missing": True,
            "missing_requested": planted_missing,
            "mostly_empty_column": "review_code",
            "outliers": {
                column: {"rows": count, "value": value}
                for column, (count, value) in outlier_plan.items()
            },
            "exact_copies": 100,
            "conflicting_label_copies": 50,
            "noisy_label_flips": 50,
            "synthetic_id": "record_id",
        },
        "observed": _observed_evidence(
            messy, "income", "income_copy", "record_id"
        ),
    }
    return messy, metadata


def build_creditcard() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(ORIGINAL_DIR / "creditcard.csv")
    df = source.copy()
    df.insert(
        0, "transaction_id", _stable_source_ids(source, "TXN")
    )
    rng = np.random.default_rng(SEED + 2)

    planted_missing = {
        "V1": _set_missing(df, rng, "V1", 0.02),
        "V14": _set_missing(df, rng, "V14", 0.03),
        "Amount": _set_missing(df, rng, "Amount", 0.04),
    }
    _add_mostly_empty_column(
        df, rng, "device_note", ["MOBILE", "DESKTOP", "REVIEW"]
    )

    used: set[int] = set()
    outlier_plan = {
        "Amount": (120, 9999999),
        "V1": (100, -999),
        "V14": (100, -999),
    }
    for column, (count, value) in outlier_plan.items():
        chosen = _choose(rng, df.index, count, used)
        df.loc[chosen, column] = value

    messy = _finish_dataset(
        df=df,
        rng=rng,
        target="Class",
        target_copy="ClassCopy",
        flip_mapping={0: 1, 1: 0},
        exact_count=200,
        conflict_count=80,
        noisy_label_count=40,
        used=used,
    )
    metadata = {
        "source_file": "Original/creditcard.csv",
        "output_file": "Messy/creditcard_messy.csv",
        "target": "Class",
        "id_column": "transaction_id",
        "target_copy": "ClassCopy",
        "source_rows": int(len(source)),
        "source_exact_duplicate_extras": int(
            source.duplicated().sum()
        ),
        "planted": {
            "missing_requested": planted_missing,
            "mostly_empty_column": "device_note",
            "outliers": {
                column: {"rows": count, "value": value}
                for column, (count, value) in outlier_plan.items()
            },
            "exact_copies": 200,
            "conflicting_label_copies": 80,
            "noisy_label_flips": 40,
            "synthetic_id": "transaction_id",
        },
        "observed": _observed_evidence(
            messy, "Class", "ClassCopy", "transaction_id"
        ),
    }
    return messy, metadata


def main() -> None:
    MESSY_DIR.mkdir(parents=True, exist_ok=True)
    builders: list[
        tuple[str, Callable[[], tuple[pd.DataFrame, dict[str, Any]]]]
    ] = [
        ("telco_messy.csv", build_telco),
        ("adult_messy.csv", build_adult),
        ("creditcard_messy.csv", build_creditcard),
    ]
    manifest: dict[str, Any] = {"seed": SEED, "datasets": {}}

    for filename, builder in builders:
        messy, metadata = builder()
        output_path = MESSY_DIR / filename
        messy.to_csv(output_path, index=False)
        manifest["datasets"][filename] = metadata
        print(
            f"Wrote {output_path.relative_to(ROOT.parent)} "
            f"({len(messy):,} rows, {len(messy.columns)} columns)"
        )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
