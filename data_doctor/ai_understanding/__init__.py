"""AI-Powered Data Understanding Engine for DataDoctor (V2).

Independent semantic interpretation layer for classification datasets.
"""

from __future__ import annotations

import datetime
import logging
from typing import Callable
import pandas as pd

from data_doctor.ai_understanding.ai_client import (
    AIClient,
    AIClientError,
    APIKeyMissingError,
    GeminiAIClient,
    MockAIClient,
)
from data_doctor.ai_understanding.cache_manager import (
    AICacheManager,
    compute_context_fingerprint,
)
from data_doctor.ai_understanding.column_analyzer import analyze_columns_batched
from data_doctor.ai_understanding.dataset_analyzer import analyze_dataset
from data_doctor.ai_understanding.dataset_context import (
    ColumnProfile,
    DatasetContext,
    extract_dataset_context,
)
from data_doctor.ai_understanding.response_schema import (
    AIUnderstandingResult,
    ColumnUnderstanding,
    ConfidenceLevel,
    ConsistencyStatus,
    DatasetObservation,
    DatasetUnderstanding,
    ExpectedStructure,
    ObservedConsistency,
    SemanticType,
    TargetAssessment,
)
from data_doctor.ai_understanding.sample_generator import (
    detect_sensitive_column,
    generate_representative_samples,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AIClient",
    "AIClientError",
    "APIKeyMissingError",
    "GeminiAIClient",
    "MockAIClient",
    "AICacheManager",
    "AIUnderstandingEngine",
    "DatasetContext",
    "ColumnProfile",
    "AIUnderstandingResult",
    "DatasetUnderstanding",
    "ColumnUnderstanding",
    "DatasetObservation",
    "TargetAssessment",
    "ExpectedStructure",
    "ObservedConsistency",
    "SemanticType",
    "ConfidenceLevel",
    "ConsistencyStatus",
    "compute_context_fingerprint",
    "extract_dataset_context",
    "detect_sensitive_column",
    "generate_representative_samples",
]


class AIUnderstandingEngine:
    """Main coordinator for the AI Data Understanding system."""

    def __init__(
        self,
        client: AIClient | None = None,
        cache_manager: AICacheManager | None = None,
    ) -> None:
        self.cache_manager = cache_manager or AICacheManager()
        self._custom_client = client

    def get_client(self) -> AIClient:
        """Resolve the active AI client.

        Tries GeminiAIClient with resolved API key; falls back to MockAIClient if key is missing.
        """
        if self._custom_client:
            return self._custom_client
        try:
            return GeminiAIClient()
        except APIKeyMissingError:
            logger.info("GEMINI_API_KEY not found. Using MockAIClient.")
            return MockAIClient()

    def run_understanding(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        user_description: str | None = None,
        excluded_columns: set[str] | list[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        force_refresh: bool = False,
    ) -> AIUnderstandingResult:
        """Run complete AI Data Understanding analysis.

        Prepares compact context, checks fingerprint cache, and coordinates
        dataset and batched column analysis.
        """
        client = self.get_client()
        model_name = getattr(client, "model", "mock-model")

        # 1. Extract context
        context = extract_dataset_context(
            df,
            target=target,
            user_description=user_description,
            excluded_columns=excluded_columns,
        )

        # 2. Compute fingerprint & check cache
        fingerprint = compute_context_fingerprint(context, model_name=model_name)
        if not force_refresh:
            cached = self.cache_manager.get(fingerprint)
            if cached:
                return cached

        # 3. Analyze dataset-level understanding
        dataset_understanding = analyze_dataset(context, client)

        # 4. Analyze column-level understanding in batches
        column_results = analyze_columns_batched(
            context,
            client,
            dataset_summary=dataset_understanding.description,
            progress_callback=progress_callback,
        )

        # 5. Assemble final result
        result = AIUnderstandingResult(
            dataset_understanding=dataset_understanding,
            columns=column_results,
            metadata={
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": model_name,
                "fingerprint": fingerprint,
                "target": target,
                "user_description": user_description,
                "row_count": len(df),
                "column_count": len(df.columns),
                "excluded_columns": sorted(context.excluded_columns),
                "analyzed_columns": [
                    c for c in context.column_names if c not in context.excluded_columns
                ],
            },
        )

        # 6. Store in cache
        self.cache_manager.put(fingerprint, result)
        return result
