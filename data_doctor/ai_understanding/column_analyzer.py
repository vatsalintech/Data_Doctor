"""Column-level analyzer module.

Orchestrates batched column analysis with shared dataset context to avoid
context limit blowouts on wide classification datasets.
"""

from __future__ import annotations

import logging
from typing import Callable
from data_doctor.ai_understanding.ai_client import AIClient
from data_doctor.ai_understanding.dataset_context import DatasetContext
from data_doctor.ai_understanding.prompts import (
    SYSTEM_INSTRUCTION,
    build_column_batch_prompt,
)
from data_doctor.ai_understanding.response_schema import (
    ColumnUnderstanding,
    ExpectedStructure,
    ObservedConsistency,
)
from data_doctor.ai_understanding.result_validator import validate_column_batch

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 5


def analyze_columns_batched(
    context: DatasetContext,
    client: AIClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dataset_summary: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, ColumnUnderstanding]:
    """Analyze all columns in manageable batches.

    Guarantees every column receives an entry even if a specific batch fails.
    """
    results: dict[str, ColumnUnderstanding] = {}
    columns_to_analyze = list(context.column_names)
    total_columns = len(columns_to_analyze)

    # Chunk columns into batches
    batches = [
        columns_to_analyze[i : i + batch_size]
        for i in range(0, total_columns, batch_size)
    ]

    for batch_idx, batch_cols in enumerate(batches):
        if progress_callback:
            progress_callback(len(results), total_columns)

        prompt = build_column_batch_prompt(
            context,
            batch_cols,
            dataset_summary=dataset_summary,
        )

        try:
            raw_response = client.generate_json(
                prompt,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            batch_result = validate_column_batch(raw_response, batch_cols)
            results.update(batch_result)
        except Exception as err:
            logger.error(
                "Batch %d (%s) failed with error: %s",
                batch_idx + 1,
                batch_cols,
                err,
            )
            # Create fallback entries for these columns so analysis doesn't completely halt
            for col in batch_cols:
                results[col] = ColumnUnderstanding(
                    column_name=col,
                    likely_meaning=f"Attribute '{col}'.",
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
                            f"Analysis batch encountered an error: {err}"
                        ],
                    ),
                    confidence="low",
                    limitations=["Batch processing was interrupted for this column."],
                    observations=[],
                )

    if progress_callback:
        progress_callback(total_columns, total_columns)

    return results
