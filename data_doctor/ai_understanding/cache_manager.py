"""Cache management and context fingerprinting for AI Data Understanding.

Provides deterministic SHA-256 fingerprinting to detect when dataset changes,
target changes, or descriptions change, ensuring stale results are flagged
and expensive AI calls are cached safely.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from data_doctor.ai_understanding.dataset_context import DatasetContext
from data_doctor.ai_understanding.response_schema import AIUnderstandingResult

SCHEMA_VERSION = "2.0.0"


def compute_context_fingerprint(
    context: DatasetContext,
    model_name: str,
    *,
    include_target: bool = True,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a dataset context."""
    # Build a stable normalized dictionary of profiling facts and samples
    fingerprint_dict: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "row_count": context.row_count,
        "column_count": context.column_count,
        "column_names": sorted(context.column_names),
        "dtypes": {k: context.dtypes[k] for k in sorted(context.dtypes.keys())},
        "user_description": context.user_description or "",
        "excluded_columns": sorted(context.excluded_columns),
    }

    if include_target:
        fingerprint_dict["target"] = context.target or ""

    # Include column profiles
    col_summaries: dict[str, Any] = {}
    for col_name in sorted(context.columns.keys()):
        p = context.columns[col_name]
        col_summaries[col_name] = {
            "dtype": p.dtype,
            "missing_pct": p.missing_pct,
            "unique_count": p.unique_count,
            "min_val": p.min_val,
            "max_val": p.max_val,
            "median_val": p.median_val,
            "mean_val": p.mean_val,
            "common_categories": p.common_categories,
            "samples": [str(s) for s in p.representative_samples],
            "is_excluded": p.is_excluded,
        }
    fingerprint_dict["columns"] = col_summaries

    payload = json.dumps(fingerprint_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AICacheManager:
    """Manages cached AI understanding results in memory or session state."""

    def __init__(self) -> None:
        self._cache: dict[str, AIUnderstandingResult] = {}

    def get(
        self,
        fingerprint: str,
    ) -> AIUnderstandingResult | None:
        """Retrieve cached result matching the exact context fingerprint."""
        return self._cache.get(fingerprint)

    def put(
        self,
        fingerprint: str,
        result: AIUnderstandingResult,
    ) -> None:
        """Store validated AI understanding result."""
        self._cache[fingerprint] = result

    def is_stale(
        self,
        cached_fingerprint: str | None,
        current_fingerprint: str,
    ) -> bool:
        """Check if cached result fingerprint differs from current context."""
        if not cached_fingerprint:
            return True
        return cached_fingerprint != current_fingerprint

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
