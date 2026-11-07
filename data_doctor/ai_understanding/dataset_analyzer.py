"""Dataset-level analyzer module.

Orchestrates prompt generation, model invocation, and validation for
overall dataset interpretation.
"""

from __future__ import annotations

import logging
from data_doctor.ai_understanding.ai_client import AIClient
from data_doctor.ai_understanding.dataset_context import DatasetContext
from data_doctor.ai_understanding.prompts import SYSTEM_INSTRUCTION, build_dataset_prompt
from data_doctor.ai_understanding.response_schema import DatasetUnderstanding
from data_doctor.ai_understanding.result_validator import validate_dataset_understanding

logger = logging.getLogger(__name__)


def analyze_dataset(context: DatasetContext, client: AIClient) -> DatasetUnderstanding:
    """Perform dataset-level semantic interpretation using the AI client."""
    prompt = build_dataset_prompt(context)
    raw_response = client.generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
    return validate_dataset_understanding(raw_response)
