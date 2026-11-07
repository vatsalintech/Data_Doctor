"""Provider-independent AI client interface with Gemini REST and Mock implementations.

Handles API authentication, timeouts, rate limits, retry backoff, and JSON parsing.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
import requests


class AIClientError(Exception):
    """Base exception for all AI client operations."""


class APIKeyMissingError(AIClientError):
    """Raised when the required API key is not configured."""


class APIRateLimitError(AIClientError):
    """Raised when rate limits or quotas are exceeded."""


class APIConnectionError(AIClientError):
    """Raised on connection timeout or network failure."""


class InvalidJSONResponseError(AIClientError):
    """Raised when the AI model returns malformed non-JSON output."""


class AIClient:
    """Abstract base class for AI provider implementations."""

    def generate_json(
        self, prompt: str, system_instruction: str = ""
    ) -> dict[str, Any]:
        """Generate structured JSON response for a prompt and system instruction."""
        raise NotImplementedError


class GeminiAIClient(AIClient):
    """Google Gemini client using direct REST API requests."""

    DEFAULT_MODEL = "gemini-flash-lite-latest"
    # Tried in order when the configured model is busy, missing, or too slow.
    FALLBACK_MODELS = (
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    )
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 1,
    ) -> None:
        self.api_key = api_key or self._resolve_api_key()
        if not self.api_key:
            raise APIKeyMissingError(
                "Gemini API key is not configured. Set the GEMINI_API_KEY environment "
                "variable or add it to .streamlit/secrets.toml."
            )
        self.model = model or self._resolve_model(self.DEFAULT_MODEL)
        self.configured_model = self.model
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds != 60 else self._resolve_timeout(60)
        )
        self.max_retries = max_retries

    @staticmethod
    def _resolve_api_key() -> str | None:
        """Resolve API key from environment variable or Streamlit secrets."""
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key.strip()
        try:
            import streamlit as st

            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _resolve_model(default_model: str) -> str:
        """Resolve model name from environment variable or Streamlit secrets."""
        model = os.environ.get("GEMINI_MODEL")
        if model:
            return model.strip()
        try:
            import streamlit as st

            if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
                return str(st.secrets["GEMINI_MODEL"]).strip()
        except Exception:
            pass
        return default_model

    @staticmethod
    def _resolve_timeout(default_timeout: int = 60) -> int:
        """Resolve request timeout from environment variable or Streamlit secrets."""
        val = os.environ.get("GEMINI_TIMEOUT")
        if val:
            try:
                return int(val)
            except ValueError:
                pass
        try:
            import streamlit as st

            if hasattr(st, "secrets") and "GEMINI_TIMEOUT" in st.secrets:
                return int(st.secrets["GEMINI_TIMEOUT"])
        except Exception:
            pass
        return default_timeout

    def _models_to_try(self) -> list[str]:
        """Configured model first, then lighter models that are more likely to answer."""
        models = [self.configured_model]
        for name in self.FALLBACK_MODELS:
            if name not in models:
                models.append(name)
        return models

    def _read_timeout(self, model_name: str, models: list[str]) -> float:
        """Give backup models a full wait. Abandon a busy model sooner."""
        full_wait = float(self.timeout_seconds)
        if model_name == models[-1]:
            return full_wait
        return min(full_wait, 20.0)

    def generate_json(
        self, prompt: str, system_instruction: str = ""
    ) -> dict[str, Any]:
        """Call Gemini REST API requesting strict JSON output.

        A busy or missing configured model is not retried until the full timeout.
        The same prompt is sent to the next lighter model instead.
        """
        models = self._models_to_try()
        last_error: Exception | None = None
        for model_name in models:
            try:
                result = self._generate_once(
                    model_name,
                    prompt,
                    system_instruction,
                    read_timeout=self._read_timeout(model_name, models),
                )
            except (APIConnectionError, APIRateLimitError) as err:
                last_error = err
                continue
            except AIClientError as err:
                if "was not found" in str(err):
                    last_error = err
                    continue
                raise
            self.model = model_name
            return result

        configured = self.configured_model
        detail = f" Last error: {last_error}" if last_error else ""
        raise APIConnectionError(
            f"Gemini did not respond. The configured model '{configured}' was "
            f"busy or too slow, and the backup models did not answer either.{detail}"
        )

    def _generate_once(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str,
        read_timeout: float,
    ) -> dict[str, Any]:
        """Send one generateContent request and parse the JSON body."""
        endpoint = f"{self.BASE_URL}/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 4096,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    params={"key": self.api_key},
                    json=payload,
                    headers=headers,
                    timeout=(10, read_timeout),
                )
            except requests.Timeout:
                last_error = APIConnectionError(
                    f"Gemini model '{model_name}' timed out after {int(read_timeout)} seconds."
                )
                continue
            except requests.RequestException as req_err:
                last_error = APIConnectionError(
                    f"Network error communicating with Gemini API: {req_err}"
                )
                continue

            if response.status_code == 200:
                return self._parse_json_response(response.json())

            if response.status_code == 429:
                last_error = APIRateLimitError(
                    f"Gemini API rate limit exceeded (HTTP 429) for model '{model_name}'."
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
                continue

            if response.status_code in {500, 502, 503, 504}:
                last_error = APIConnectionError(
                    f"Gemini model '{model_name}' is temporarily unavailable "
                    f"(HTTP {response.status_code})."
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
                continue

            if response.status_code == 404:
                raise AIClientError(
                    f"Gemini model '{model_name}' was not found (HTTP 404)."
                )

            raise AIClientError(
                f"Gemini API error (HTTP {response.status_code}): {response.text}"
            )

        if last_error:
            raise last_error
        raise AIClientError("Failed to obtain a valid response from Gemini API.")

    @staticmethod
    def _parse_json_response(data: dict[str, Any]) -> dict[str, Any]:
        """Pull the JSON object out of a generateContent response."""
        candidates = data.get("candidates", [])
        if not candidates:
            raise InvalidJSONResponseError(
                "Gemini returned an empty candidate list (content may be filtered)."
            )
        content_parts = candidates[0].get("content", {}).get("parts", [])
        if not content_parts:
            raise InvalidJSONResponseError("Gemini candidate has no content parts.")
        raw_text = ""
        for part in content_parts:
            if part.get("thought"):
                continue
            text = part.get("text", "")
            if text.strip():
                raw_text = text.strip()
                break
        if not raw_text:
            raw_text = content_parts[0].get("text", "").strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as decode_err:
            raise InvalidJSONResponseError(
                f"Model response was not valid JSON: {decode_err}\nRaw text: {raw_text[:200]}"
            ) from decode_err
        if not isinstance(parsed, dict):
            raise InvalidJSONResponseError("Model response JSON was not an object.")
        return parsed


class MockAIClient(AIClient):
    """Deterministic mock AI client for test suites and offline operation."""

    def __init__(
        self,
        should_fail: bool = False,
        failure_type: str = "error",
        custom_responses: dict[str, Any] | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.failure_type = failure_type
        self.custom_responses = custom_responses or {}
        self.call_count = 0
        self.history: list[dict[str, str]] = []

    def generate_json(
        self, prompt: str, system_instruction: str = ""
    ) -> dict[str, Any]:
        self.call_count += 1
        self.history.append({"prompt": prompt, "system_instruction": system_instruction})

        if self.should_fail:
            if self.failure_type == "rate_limit":
                raise APIRateLimitError("Mock rate limit exceeded")
            if self.failure_type == "timeout":
                raise APIConnectionError("Mock timeout exceeded")
            if self.failure_type == "invalid_json":
                raise InvalidJSONResponseError("Mock returned invalid JSON")
            raise AIClientError("Mock client generic error")

        # Check custom responses
        for key, res in self.custom_responses.items():
            if key in prompt:
                return res

        # Check if dataset understanding prompt
        if "REQUIRED JSON SCHEMA" in prompt and "composition" in prompt:
            return self._mock_dataset_response(prompt)

        # Batch column prompt
        if "COLUMNS TO INTERPRET" in prompt or "columns" in prompt:
            return self._mock_column_batch_response(prompt)

        # Fallback
        return {"status": "ok", "message": "Generic mock response"}

    def _mock_dataset_response(self, prompt: str) -> dict[str, Any]:
        """Synthesize mock dataset-level understanding based on keywords."""
        is_telco = "Churn" in prompt or "TotalCharges" in prompt or "MonthlyCharges" in prompt
        is_adult = "income" in prompt or "education" in prompt or "hours-per-week" in prompt
        is_credit = "Class" in prompt or "V1" in prompt or "Amount" in prompt

        if is_telco:
            return {
                "description": "The dataset appears to contain telecommunications customer subscription, demographic, and account billing information.",
                "possible_purpose": "Customer churn prediction and retention risk modeling.",
                "composition": [
                    "Customer Identifiers",
                    "Demographic Details",
                    "Service Subscriptions",
                    "Billing and Contract Information",
                    "Churn Outcome",
                ],
                "observations": [
                    {
                        "title": "Related billing concepts",
                        "description": "MonthlyCharges and TotalCharges appear to represent related billing amounts across customer tenure.",
                        "confidence": "high",
                        "limitation": "The exact billing formula cannot be verified without business contract rules.",
                    },
                    {
                        "title": "Contract and tenure relationship",
                        "description": "Tenure and Contract type may correlate with customer retention behavior.",
                        "confidence": "medium",
                        "limitation": "Causal effects cannot be deduced from static tabular snapshot.",
                    },
                ],
                "confidence": "high",
                "limitations": [
                    "Currency units are not explicitly labeled in the column metadata.",
                    "Timeframe of customer records is not stated.",
                ],
                "target_assessment": {
                    "target_name": "Churn",
                    "target_meaning": "Binary indicator of whether the customer discontinued service.",
                    "consistency_with_purpose": "Consistent with customer churn risk prediction.",
                    "observations": [
                        "Target is binary and aligns with telecommunications retention modeling."
                    ],
                }
                if "Churn" in prompt
                else None,
            }

        if is_adult:
            return {
                "description": "The dataset appears to contain demographic, employment, and census information for adult individuals.",
                "possible_purpose": "Income bracket prediction (e.g., predicting whether annual income exceeds $50K).",
                "composition": [
                    "Demographic Attributes",
                    "Education Records",
                    "Employment and Occupation Information",
                    "Financial Capital Indicators",
                    "Income Threshold Outcome",
                ],
                "observations": [
                    {
                        "title": "Education and educational level overlap",
                        "description": "education and education-num represent equivalent educational attainment concepts in categorical and ordinal formats.",
                        "confidence": "high",
                        "limitation": "One column is likely an integer coding of the other.",
                    }
                ],
                "confidence": "high",
                "limitations": [
                    "Historical census data may reflect socioeconomic patterns from a specific era."
                ],
                "target_assessment": {
                    "target_name": "income",
                    "target_meaning": "Categorical threshold indicator for individual income.",
                    "consistency_with_purpose": "Directly matches binary census income classification benchmarks.",
                    "observations": [
                        "Imbalance exists between majority <=50K and minority >50K."
                    ],
                }
                if "income" in prompt
                else None,
            }

        if is_credit:
            return {
                "description": "The dataset appears to contain credit card transaction features, possibly PCA-transformed components, along with transaction amounts and fraud labels.",
                "possible_purpose": "Credit card fraud detection and anomaly classification.",
                "composition": [
                    "Anonymized PCA Transformed Numerical Features",
                    "Transaction Timestamp / Time Offsets",
                    "Transaction Amount",
                    "Fraud Class Indicator",
                ],
                "observations": [
                    {
                        "title": "Transformed numerical features",
                        "description": "Columns V1 through V28 appear to be principal component projections to preserve transaction privacy.",
                        "confidence": "high",
                        "limitation": "Underlying financial meanings of PCA components are concealed by design.",
                    }
                ],
                "confidence": "high",
                "limitations": [
                    "Feature names are anonymized, preventing semantic domain interpretation."
                ],
                "target_assessment": {
                    "target_name": "Class",
                    "target_meaning": "Binary indicator where 1 represents fraudulent transactions and 0 represents legitimate transactions.",
                    "consistency_with_purpose": "Directly aligns with fraud detection objectives.",
                    "observations": [
                        "Severe class imbalance is typical for transaction fraud benchmarks."
                    ],
                }
                if "Class" in prompt
                else None,
            }

        # Generic dataset response
        return {
            "description": "The dataset appears to represent structured tabular records containing mixed numerical and categorical attributes.",
            "possible_purpose": "Supervised tabular classification or predictive modeling.",
            "composition": ["Feature Attributes", "Categorical Indicators"],
            "observations": [
                {
                    "title": "Mixed tabular schema",
                    "description": "Columns contain both numeric measurements and categorical group labels.",
                    "confidence": "medium",
                    "limitation": "Domain-specific context requires user-supplied documentation.",
                }
            ],
            "confidence": "medium",
            "limitations": [
                "Dataset domain cannot be conclusively determined without additional business metadata."
            ],
            "target_assessment": None,
        }

    def _mock_column_batch_response(self, prompt: str) -> dict[str, Any]:
        """Synthesize mock column interpretations for columns requested in prompt."""
        # Find column names mentioned in prompt
        cols_interpreted: list[dict[str, Any]] = []

        # Extract column names by looking at "column_name": "..."
        matches = re.findall(r'"column_name":\s*"([^"]+)"', prompt)
        seen = set()
        for col in matches:
            if col in seen:
                continue
            seen.add(col)
            cols_interpreted.append(self._synthesize_column_understanding(col))

        if not cols_interpreted:
            cols_interpreted.append(self._synthesize_column_understanding("generic_column"))

        return {"columns": cols_interpreted}

    def _synthesize_column_understanding(self, col: str) -> dict[str, Any]:
        """Synthesize realistic column understanding for a given column name."""
        col_lower = col.lower()

        # Specific known columns
        if "customerid" in col_lower or col_lower in {"id", "uuid"}:
            return {
                "column_name": col,
                "likely_meaning": "Unique customer or entity identifier.",
                "semantic_type": "identifier",
                "secondary_semantic_types": ["identifier"],
                "expected_structure": {
                    "logical_type": "text",
                    "expected_format": "Alphanumeric code or UUID",
                    "expected_constraints": ["Unique per record"],
                    "expected_units": None,
                    "possible_range": None,
                },
                "observed_consistency": {
                    "status": "consistent",
                    "observations": [
                        "Sample values appear to be unique alphanumeric codes."
                    ],
                },
                "confidence": "high",
                "limitations": [
                    "Identifier format is specific to the source system."
                ],
                "observations": ["Typically should not be used as a predictive feature."],
            }

        if "churn" in col_lower or "fraud" in col_lower or col_lower in {"class", "target", "label"}:
            return {
                "column_name": col,
                "likely_meaning": "Target outcome indicating event occurrence or class label.",
                "semantic_type": "binary_indicator",
                "secondary_semantic_types": ["nominal_categorical"],
                "expected_structure": {
                    "logical_type": "categorical",
                    "expected_format": "Binary indicator (Yes/No or 0/1)",
                    "expected_constraints": ["Two distinct categories"],
                    "expected_units": None,
                    "possible_range": None,
                },
                "observed_consistency": {
                    "status": "consistent",
                    "observations": [
                        "Sample values conform to binary outcome states."
                    ],
                },
                "confidence": "high",
                "limitations": ["True class definition depends on organizational criteria."],
                "observations": ["Serves as the prediction target."],
            }

        if "charges" in col_lower or "amount" in col_lower or "income" in col_lower:
            return {
                "column_name": col,
                "likely_meaning": "Monetary billing or transaction amount.",
                "semantic_type": "monetary",
                "secondary_semantic_types": ["continuous_numeric"],
                "expected_structure": {
                    "logical_type": "numerical",
                    "expected_format": "Decimal or floating point currency values",
                    "expected_constraints": ["Non-negative monetary values"],
                    "expected_units": "Currency (USD or local)",
                    "possible_range": "Zero or positive amounts",
                },
                "observed_consistency": {
                    "status": "consistent",
                    "observations": [
                        "Sample values represent non-negative numerical amounts."
                    ],
                },
                "confidence": "high",
                "limitations": [
                    "Specific currency symbol is not embedded in the data."
                ],
                "observations": ["Reflects monetary scale."],
            }

        if "tenure" in col_lower or "age" in col_lower or "time" in col_lower:
            return {
                "column_name": col,
                "likely_meaning": "Duration, elapsed time period, or age in discrete/continuous units.",
                "semantic_type": "continuous_numeric",
                "secondary_semantic_types": ["discrete_numeric"],
                "expected_structure": {
                    "logical_type": "numerical",
                    "expected_format": "Integer or floating point duration count",
                    "expected_constraints": ["Non-negative values"],
                    "expected_units": "Months, years, or seconds depending on context",
                    "possible_range": "Zero or greater",
                },
                "observed_consistency": {
                    "status": "consistent",
                    "observations": [
                        "Values represent sequential or elapsed counts."
                    ],
                },
                "confidence": "medium",
                "limitations": [
                    "Exact time unit requires source documentation."
                ],
                "observations": ["Temporal measurement."],
            }

        if col.startswith("V") and col[1:].isdigit():
            return {
                "column_name": col,
                "likely_meaning": f"PCA-transformed numerical component {col}.",
                "semantic_type": "continuous_numeric",
                "secondary_semantic_types": ["continuous_numeric"],
                "expected_structure": {
                    "logical_type": "numerical",
                    "expected_format": "Zero-centered continuous floating point values",
                    "expected_constraints": ["Real-valued numbers"],
                    "expected_units": None,
                    "possible_range": "Unbounded real numbers",
                },
                "observed_consistency": {
                    "status": "consistent",
                    "observations": [
                        "Values align with standardized principal component distributions."
                    ],
                },
                "confidence": "high",
                "limitations": [
                    "Original underlying financial metric was anonymized."
                ],
                "observations": ["Anonymized feature representation."],
            }

        # General default
        return {
            "column_name": col,
            "likely_meaning": f"Attribute representing {col.replace('_', ' ').title()}.",
            "semantic_type": "nominal_categorical"
            if "gender" in col_lower or "contract" in col_lower or "education" in col_lower
            else "continuous_numeric",
            "secondary_semantic_types": [],
            "expected_structure": {
                "logical_type": "categorical"
                if "gender" in col_lower or "contract" in col_lower
                else "numerical",
                "expected_format": None,
                "expected_constraints": [],
                "expected_units": None,
                "possible_range": None,
            },
            "observed_consistency": {
                "status": "consistent",
                "observations": [
                    "Supplied values match the expected format."
                ],
            },
            "confidence": "medium",
            "limitations": ["Requires domain context for further precision."],
            "observations": [],
        }
