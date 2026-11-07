"""Representative sample generator and sensitive-column heuristics.

Safely prepares 15-20 representative non-missing values per column without
sending the entire dataset to external models. Detects potential identifiers
and PII columns to recommend exclusion by default.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 18
DEFAULT_SEED = 42

SENSITIVE_NAME_PATTERNS = [
    r"(?i)\b(ssn|social_security|socialsecurity)\b",
    r"(?i)\b(credit_card|creditcard|card_num|card_number|cvv|pan)\b",
    r"(?i)\b(passport|license|driver_license)\b",
    r"(?i)\b(email|e_mail|mail_address)\b",
    r"(?i)\b(phone|telephone|mobile|cell_phone|fax)\b",
    r"(?i)\b(first_name|last_name|full_name|surname|customer_name|client_name|patient_name)\b",
    r"(?i)\b(address|street|apartment|zip_code|postal_code)\b",
    r"(?i)\b(password|secret|token|api_key|auth)\b",
]

ID_NAME_PATTERNS = [
    r"(?i)\b(id|uuid|guid|customer_id|customerid|user_id|userid|account_id|accountid|client_id|patient_id)\b",
    r"(?i)^id$|_id$|^id_|_id_",
]

EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
PHONE_REGEX = re.compile(r"^\+?[\d\s\-\(\)]{7,20}$")
UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def detect_sensitive_column(
    series: pd.Series, column_name: str
) -> tuple[bool, str | None]:
    """Heuristically detect if a column is likely an identifier or contains sensitive data.

    Returns (is_sensitive, reason).
    """
    col_str = str(column_name).strip()

    # 1. Match sensitive name patterns
    for pattern in SENSITIVE_NAME_PATTERNS:
        if re.search(pattern, col_str):
            return True, f"Column name matches sensitive pattern ({pattern})"

    # 2. Match ID name patterns
    for pattern in ID_NAME_PATTERNS:
        if re.search(pattern, col_str):
            return True, "Column name suggests an identifier or account key"

    # 3. Check sample values for string series
    non_null = series.dropna()
    if len(non_null) == 0:
        return False, None

    # Inspect a small sample of string values
    sample_values = [str(v).strip() for v in non_null.head(50)]

    email_matches = sum(bool(EMAIL_REGEX.match(v)) for v in sample_values)
    if email_matches > len(sample_values) * 0.5:
        return True, "Column content contains email addresses"

    phone_matches = sum(bool(PHONE_REGEX.match(v)) for v in sample_values)
    if phone_matches > len(sample_values) * 0.7 and not pd.api.types.is_numeric_dtype(series):
        return True, "Column content appears to contain phone numbers"

    uuid_matches = sum(bool(UUID_REGEX.match(v)) for v in sample_values)
    if uuid_matches > len(sample_values) * 0.5:
        return True, "Column content contains UUIDs or machine identifiers"

    # High-cardinality non-numeric unique ratio (>98% unique with >= 30 rows)
    if not pd.api.types.is_numeric_dtype(series) and len(non_null) >= 30:
        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio > 0.98:
            return True, f"High cardinality unique text values ({unique_ratio:.1%} unique)"

    return False, None


def sanitize_sample_value(value: Any) -> Any:
    """Format and sanitize a single cell value for safe AI prompt inclusion.

    Truncates overly long strings, avoids nan/inf, and standardizes types.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        if np.isneginf(value) or np.isposinf(value) or np.isnan(value):
            return str(value)
        # Clean rounding for readability
        return round(float(value), 4) if abs(value) < 1e6 else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)

    # String sanitization
    val_str = str(value).strip()
    # Strip potential control characters
    val_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", val_str)
    # Truncate overly long values to avoid token blowout or prompt injection
    if len(val_str) > 100:
        val_str = val_str[:97] + "..."
    return val_str


def generate_representative_samples(
    series: pd.Series,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Any]:
    """Generate 15-20 representative non-missing values from a column.

    Sampling strategy:
    - ~3-4 from the beginning
    - ~3-4 from the middle
    - ~3-4 from the end
    - Random remainder with fixed seed for reproducibility
    - Deduplicates while preserving discovery order
    - If non-null values <= sample_size, returns all available unique non-nulls.
    """
    non_null = series.dropna()
    total_valid = len(non_null)

    if total_valid == 0:
        return []

    # If small count, return all sanitized unique values
    if total_valid <= sample_size:
        seen = set()
        result = []
        for val in non_null:
            sanitized = sanitize_sample_value(val)
            if sanitized is not None and str(sanitized) not in seen:
                seen.add(str(sanitized))
                result.append(sanitized)
        return result

    # Divide sampling across beginning, middle, end, and random
    n_head = min(4, total_valid)
    n_tail = min(4, total_valid)
    n_mid = min(4, total_valid)

    head_indices = list(range(0, n_head))
    mid_start = max(n_head, (total_valid // 2) - (n_mid // 2))
    mid_indices = list(range(mid_start, min(total_valid, mid_start + n_mid)))
    tail_indices = list(range(max(0, total_valid - n_tail), total_valid))

    collected_indices = set(head_indices + mid_indices + tail_indices)

    # Add random indices to reach desired sample size
    remaining_indices = [i for i in range(total_valid) if i not in collected_indices]
    needed_random = max(0, sample_size - len(collected_indices))
    if needed_random > 0 and remaining_indices:
        rng = np.random.default_rng(seed)
        chosen_random = rng.choice(
            remaining_indices,
            size=min(needed_random, len(remaining_indices)),
            replace=False,
        ).tolist()
        collected_indices.update(chosen_random)

    sorted_indices = sorted(collected_indices)

    seen = set()
    samples: list[Any] = []
    for idx in sorted_indices:
        val = non_null.iloc[idx]
        sanitized = sanitize_sample_value(val)
        val_key = str(sanitized)
        if sanitized is not None and val_key not in seen:
            seen.add(val_key)
            samples.append(sanitized)
        if len(samples) >= sample_size:
            break

    # If deduplication dropped sample count too low, top up from remaining unique values
    if len(samples) < min(15, total_valid):
        for val in non_null:
            sanitized = sanitize_sample_value(val)
            val_key = str(sanitized)
            if sanitized is not None and val_key not in seen:
                seen.add(val_key)
                samples.append(sanitized)
            if len(samples) >= sample_size:
                break

    return samples
