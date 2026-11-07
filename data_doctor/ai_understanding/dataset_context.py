"""Dataset context preparation module.

Extracts compact, accurate Python/pandas metadata and statistical profiles for
feeding to the AI engine without sending full raw datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from data_doctor.ai_understanding.sample_generator import (
    detect_sensitive_column,
    generate_representative_samples,
)


@dataclass
class ColumnProfile:
    """Statistical and profiling summary of a single column."""

    name: str
    dtype: str
    profile_type: str  # "numeric", "categorical", "datetime", "text", "empty"
    missing_count: int
    missing_pct: float
    unique_count: int
    min_val: float | None = None
    max_val: float | None = None
    median_val: float | None = None
    mean_val: float | None = None
    common_categories: list[dict[str, Any]] = field(default_factory=list)
    representative_samples: list[Any] = field(default_factory=list)
    is_sensitive_candidate: bool = False
    exclusion_reason: str | None = None
    is_excluded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetContext:
    """Compact context representation of a dataset for AI analysis."""

    row_count: int
    column_count: int
    column_names: list[str]
    dtypes: dict[str, str]
    target: str | None = None
    user_description: str | None = None
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    excluded_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "dtypes": self.dtypes,
            "target": self.target,
            "user_description": self.user_description,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "excluded_columns": self.excluded_columns,
        }


def profile_column(
    series: pd.Series,
    column_name: str,
    *,
    is_excluded: bool = False,
    sample_seed: int = 42,
) -> ColumnProfile:
    """Calculate statistical summaries and representative samples for a column."""
    name = str(column_name)
    dtype_str = str(series.dtype)
    total_len = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = round(float(missing_count / total_len * 100), 2) if total_len > 0 else 0.0

    non_null = series.dropna()
    unique_count = int(non_null.nunique())

    is_sensitive, sensitive_reason = detect_sensitive_column(series, name)

    # Empty column
    if len(non_null) == 0:
        return ColumnProfile(
            name=name,
            dtype=dtype_str,
            profile_type="empty",
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=0,
            is_sensitive_candidate=is_sensitive,
            exclusion_reason=sensitive_reason,
            is_excluded=is_excluded,
        )

    # Datetime column
    if pd.api.types.is_datetime64_any_dtype(series):
        samples = (
            []
            if is_excluded
            else generate_representative_samples(series, seed=sample_seed)
        )
        return ColumnProfile(
            name=name,
            dtype=dtype_str,
            profile_type="datetime",
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            representative_samples=samples,
            is_sensitive_candidate=is_sensitive,
            exclusion_reason=sensitive_reason,
            is_excluded=is_excluded,
        )

    # Numeric column
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Clean finite values for stats
        finite_vals = non_null[np.isfinite(non_null)]
        min_v = float(finite_vals.min()) if len(finite_vals) > 0 else None
        max_v = float(finite_vals.max()) if len(finite_vals) > 0 else None
        median_v = float(finite_vals.median()) if len(finite_vals) > 0 else None
        mean_v = float(finite_vals.mean()) if len(finite_vals) > 0 else None

        # Check if actually behaves as binary indicator (e.g. {0, 1})
        is_binary_num = unique_count == 2 and set(non_null.unique()).issubset({0, 1, 0.0, 1.0})
        profile_type = "numeric" if not is_binary_num else "categorical"

        common_cats: list[dict[str, Any]] = []
        if is_binary_num or unique_count <= 10:
            top_counts = non_null.value_counts().head(5)
            common_cats = [
                {"category": str(k), "count": int(v), "percentage": round(float(v / total_len * 100), 2)}
                for k, v in top_counts.items()
            ]

        samples = (
            []
            if is_excluded
            else generate_representative_samples(series, seed=sample_seed)
        )

        return ColumnProfile(
            name=name,
            dtype=dtype_str,
            profile_type=profile_type,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            min_val=min_v,
            max_val=max_v,
            median_val=median_v,
            mean_val=mean_v,
            common_categories=common_cats,
            representative_samples=samples,
            is_sensitive_candidate=is_sensitive,
            exclusion_reason=sensitive_reason,
            is_excluded=is_excluded,
        )

    # Categorical or string / object column
    top_counts = non_null.value_counts().head(8)
    common_cats = [
        {"category": str(k), "count": int(v), "percentage": round(float(v / total_len * 100), 2)}
        for k, v in top_counts.items()
    ]

    # Decide between categorical and free text
    profile_type = "categorical"
    if unique_count > 100 and unique_count / len(non_null) > 0.8:
        profile_type = "text"

    samples = (
        []
        if is_excluded
        else generate_representative_samples(series, seed=sample_seed)
    )

    return ColumnProfile(
        name=name,
        dtype=dtype_str,
        profile_type=profile_type,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        common_categories=common_cats,
        representative_samples=samples,
        is_sensitive_candidate=is_sensitive,
        exclusion_reason=sensitive_reason,
        is_excluded=is_excluded,
    )


def extract_dataset_context(
    df: pd.DataFrame,
    target: str | None = None,
    user_description: str | None = None,
    excluded_columns: set[str] | list[str] | None = None,
    sample_seed: int = 42,
) -> DatasetContext:
    """Extract complete context representation of the dataset for AI analysis.

    Calculates deterministic profiling stats and representative samples.
    Excluded columns have their samples withheld.
    """
    excluded_set = set(excluded_columns or [])
    col_profiles: dict[str, ColumnProfile] = {}

    for col in df.columns:
        is_excl = col in excluded_set
        profile = profile_column(
            df[col],
            str(col),
            is_excluded=is_excl,
            sample_seed=sample_seed,
        )
        col_profiles[str(col)] = profile

    return DatasetContext(
        row_count=len(df),
        column_count=len(df.columns),
        column_names=[str(c) for c in df.columns],
        dtypes={str(c): str(df[c].dtype) for c in df.columns},
        target=str(target) if target else None,
        user_description=user_description.strip() if user_description else None,
        columns=col_profiles,
        excluded_columns=sorted(excluded_set),
    )
