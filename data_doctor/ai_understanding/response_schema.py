"""Pydantic schemas for AI-Powered Data Understanding Engine (V2).

Defines structured output models, controlled vocabularies, and validation rules.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

ConfidenceLevel = Literal["high", "medium", "low"]

ConsistencyStatus = Literal[
    "consistent",
    "potentially_inconsistent",
    "insufficient_information",
    "ambiguous",
]

SemanticType = Literal[
    "continuous_numeric",
    "discrete_numeric",
    "binary_indicator",
    "nominal_categorical",
    "ordinal_categorical",
    "identifier",
    "datetime",
    "free_text",
    "geographic",
    "monetary",
    "percentage_ratio",
    "unknown_ambiguous",
]


class DatasetObservation(BaseModel):
    """Observation regarding cross-column relationships or dataset-level patterns."""

    title: str = Field(description="Short descriptive title of the observation.")
    description: str = Field(
        description="Factual, cautious description of the observed pattern or relationship."
    )
    confidence: ConfidenceLevel = Field(
        default="medium", description="Confidence in this specific observation."
    )
    limitation: str | None = Field(
        default=None,
        description="Contextual limitations or uncertainties regarding this observation.",
    )


class TargetAssessment(BaseModel):
    """Assessment of the selected target column in the context of the dataset."""

    target_name: str = Field(description="Name of the selected target column.")
    target_meaning: str = Field(description="What the target column appears to represent.")
    consistency_with_purpose: str = Field(
        description="Assessment of whether the target matches the apparent dataset purpose."
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Observations regarding target-feature relationships or investigation suggestions.",
    )


class DatasetUnderstanding(BaseModel):
    """Overall dataset-level interpretation produced by AI."""

    description: str = Field(
        description="Cautious summary of what the dataset appears to represent."
    )
    possible_purpose: str = Field(
        description="Possible purpose or machine learning problem the dataset is suited for."
    )
    composition: list[str] = Field(
        default_factory=list,
        description="Major categories or types of information present in the dataset.",
    )
    observations: list[DatasetObservation] = Field(
        default_factory=list,
        description="Cross-column patterns, possible relationships, or unit observations.",
    )
    confidence: ConfidenceLevel = Field(
        default="medium", description="Overall confidence level in the dataset interpretation."
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Important limitations, ambiguities, or missing business context.",
    )
    target_assessment: TargetAssessment | None = Field(
        default=None,
        description="Optional assessment of the target if one was selected.",
    )


class ExpectedStructure(BaseModel):
    """Expected semantic structure of a column based on its likely meaning."""

    logical_type: str = Field(
        description="Expected logical data type (e.g. numeric, categorical, datetime, text)."
    )
    expected_format: str | None = Field(
        default=None, description="Expected format or representation (e.g. decimal, ISO-8601)."
    )
    expected_constraints: list[str] = Field(
        default_factory=list,
        description="Expected natural constraints (e.g. non-negative, binary {0, 1}).",
    )
    expected_units: str | None = Field(
        default=None, description="Expected units if reasonably inferable (e.g. Currency, Months)."
    )
    possible_range: str | None = Field(
        default=None,
        description="Possible range without inventing strict numerical bounds.",
    )


class ObservedConsistency(BaseModel):
    """Consistency analysis comparing expected structure with actual sample values and stats."""

    status: ConsistencyStatus = Field(
        description="Status assessing alignment between expected meaning and actual values."
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Specific observations comparing expected meaning vs actual values.",
    )


class ColumnUnderstanding(BaseModel):
    """Interpretation of an individual column."""

    column_name: str = Field(description="Name of the column analyzed.")
    likely_meaning: str = Field(
        description="What the column most likely represents in plain language."
    )
    semantic_type: SemanticType = Field(
        description="Primary semantic classification from the controlled vocabulary."
    )
    secondary_semantic_types: list[str] = Field(
        default_factory=list,
        description="Optional secondary descriptors (e.g. ['monetary', 'continuous_numeric']).",
    )
    expected_structure: ExpectedStructure = Field(
        description="Expected logical structure and constraints for this column."
    )
    observed_consistency: ObservedConsistency = Field(
        description="Comparison of expected meaning against actual sample values and profiling stats."
    )
    confidence: ConfidenceLevel = Field(
        default="medium", description="Confidence in this column interpretation."
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations, ambiguities, or alternative interpretations.",
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Additional column-specific notes or nuances.",
    )


class AIUnderstandingResult(BaseModel):
    """Complete validated result combining dataset-level and column-level understanding."""

    dataset_understanding: DatasetUnderstanding
    columns: dict[str, ColumnUnderstanding] = Field(
        default_factory=dict,
        description="Map of column_name -> ColumnUnderstanding.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata: timestamp, model, context fingerprint, analyzed/excluded columns.",
    )
