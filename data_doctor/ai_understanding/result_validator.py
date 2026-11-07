"""Result validation and sanitization module.

Validates AI outputs against Pydantic schemas, verifies column name alignment,
catches partial failures, and ensures no unvalidated or malformed output reaches the UI.
"""

from __future__ import annotations

import html
import logging
from typing import Any
from pydantic import ValidationError

from data_doctor.ai_understanding.response_schema import (
    ColumnUnderstanding,
    DatasetUnderstanding,
    ExpectedStructure,
    ObservedConsistency,
)

logger = logging.getLogger(__name__)


def sanitize_text(text: Any) -> str:
    """Sanitize strings to prevent HTML/script injection in Streamlit UI."""
    if text is None:
        return ""
    raw = str(text).strip()
    return html.escape(raw)


def validate_dataset_understanding(raw_dict: dict[str, Any]) -> DatasetUnderstanding:
    """Validate raw dictionary against DatasetUnderstanding schema."""
    try:
        # Pydantic v2 validation
        return DatasetUnderstanding.model_validate(raw_dict)
    except ValidationError as err:
        logger.warning("Dataset understanding validation error: %s", err)
        # Fallback with safe defaults
        return DatasetUnderstanding(
            description=sanitize_text(raw_dict.get("description", "Dataset interpretation provided.")),
            possible_purpose=sanitize_text(
                raw_dict.get("possible_purpose", "Tabular classification task.")
            ),
            composition=[
                sanitize_text(c) for c in raw_dict.get("composition", ["Tabular features"])
            ],
            observations=[],
            confidence="low",
            limitations=[
                "AI response had minor schema inconsistencies and was normalized.",
            ],
            target_assessment=None,
        )


def validate_column_understanding(
    raw_dict: dict[str, Any],
    expected_column_name: str,
) -> ColumnUnderstanding | None:
    """Validate a single column understanding dictionary.

    Returns None if the dictionary cannot be coerced or describes a mismatched column.
    """
    col_name = str(raw_dict.get("column_name", "")).strip()
    if col_name != expected_column_name:
        # If model hallucinated or changed case, check if case-insensitive match
        if col_name.lower() == expected_column_name.lower():
            raw_dict["column_name"] = expected_column_name
        else:
            return None

    try:
        return ColumnUnderstanding.model_validate(raw_dict)
    except ValidationError as err:
        logger.warning(
            "Column understanding validation failed for column '%s': %s",
            expected_column_name,
            err,
        )
        # Attempt minimal repair
        try:
            return ColumnUnderstanding(
                column_name=expected_column_name,
                likely_meaning=sanitize_text(
                    raw_dict.get("likely_meaning", f"Attribute '{expected_column_name}'.")
                ),
                semantic_type="unknown_ambiguous",
                secondary_semantic_types=[],
                expected_structure=ExpectedStructure(
                    logical_type="unknown",
                    expected_format=None,
                    expected_constraints=[],
                    expected_units=None,
                    possible_range=None,
                ),
                observed_consistency=ObservedConsistency(
                    status="ambiguous",
                    observations=["Response required fallback schema normalization."],
                ),
                confidence="low",
                limitations=["AI model response contained non-standard format."],
                observations=[],
            )
        except Exception:
            return None


def validate_column_batch(
    raw_response: dict[str, Any],
    batch_columns: list[str],
) -> dict[str, ColumnUnderstanding]:
    """Validate a batch of column understandings against expected columns."""
    result: dict[str, ColumnUnderstanding] = {}
    raw_columns = raw_response.get("columns", [])

    if not isinstance(raw_columns, list):
        logger.warning("AI response 'columns' field was not a list.")
        raw_columns = []

    # Map raw entries by column name
    raw_by_name: dict[str, dict[str, Any]] = {}
    for item in raw_columns:
        if isinstance(item, dict) and "column_name" in item:
            raw_by_name[str(item["column_name"]).strip()] = item
            raw_by_name[str(item["column_name"]).strip().lower()] = item

    for col in batch_columns:
        item = raw_by_name.get(col) or raw_by_name.get(col.lower())
        if item:
            validated = validate_column_understanding(item, col)
            if validated:
                result[col] = validated
                continue

        # If a column was not returned or could not be validated, create an informative entry
        result[col] = ColumnUnderstanding(
            column_name=col,
            likely_meaning=f"Attribute '{col}' (analysis incomplete).",
            semantic_type="unknown_ambiguous",
            secondary_semantic_types=[],
            expected_structure=ExpectedStructure(
                logical_type="unknown",
                expected_format=None,
                expected_constraints=[],
                expected_units=None,
                possible_range=None,
            ),
            observed_consistency=ObservedConsistency(
                status="insufficient_information",
                observations=[
                    "Column interpretation was omitted or failed validation in this batch."
                ],
            ),
            confidence="low",
            limitations=["Analysis incomplete for this specific column."],
            observations=[],
        )

    return result
