"""Tests for AI-Powered Data Understanding Engine (V2)."""

import pandas as pd
import pytest

from data_doctor.ai_understanding import (
    AIUnderstandingEngine,
    ColumnProfile,
    DatasetContext,
    GeminiAIClient,
    MockAIClient,
    compute_context_fingerprint,
    detect_sensitive_column,
    extract_dataset_context,
    generate_representative_samples,
)
from data_doctor.ai_understanding.ai_client import (
    APIConnectionError,
    APIKeyMissingError,
    APIRateLimitError,
    InvalidJSONResponseError,
)
from data_doctor.ai_understanding.cache_manager import AICacheManager
from data_doctor.ai_understanding.response_schema import (
    AIUnderstandingResult,
    ColumnUnderstanding,
    DatasetUnderstanding,
    ExpectedStructure,
    ObservedConsistency,
)
from data_doctor.ai_understanding.result_validator import (
    validate_column_batch,
    validate_dataset_understanding,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(100)],
            "MonthlyCharges": [20.5 + float(i % 10) for i in range(100)],
            "tenure": [int(i % 24) for i in range(100)],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"] * 25,
            "email_address": [f"user{i}@example.com" for i in range(100)],
            "Churn": ["No", "No", "Yes", "No"] * 25,
        }
    )


# ── Sample Generator & Heuristics Tests ─────────────────────────────────────────


def test_representative_samples_size_and_reproducibility():
    s = pd.Series([float(i) for i in range(200)])
    samples1 = generate_representative_samples(s, sample_size=18, seed=42)
    samples2 = generate_representative_samples(s, sample_size=18, seed=42)

    assert len(samples1) == 18
    assert samples1 == samples2
    # Ensure head, mid, and tail values are present
    assert 0.0 in samples1
    assert 199.0 in samples1


def test_representative_samples_small_column():
    s = pd.Series(["apple", "banana", "cherry"])
    samples = generate_representative_samples(s, sample_size=18)
    assert set(samples) == {"apple", "banana", "cherry"}


def test_representative_samples_all_null():
    s = pd.Series([None, None, float("nan")])
    samples = generate_representative_samples(s)
    assert samples == []


def test_sensitive_column_detection():
    # Name patterns
    is_sens, reason = detect_sensitive_column(pd.Series(["A", "B"]), "customerID")
    assert is_sens is True
    assert "identifier" in reason.lower() or "pattern" in reason.lower()

    # Email pattern
    s_email = pd.Series([f"user{i}@test.com" for i in range(30)])
    is_sens, reason = detect_sensitive_column(s_email, "contact_info")
    assert is_sens is True
    assert "email" in reason.lower()

    # Normal numeric column should not be sensitive
    s_num = pd.Series([10.5, 20.3, 30.1, 40.2] * 10)
    is_sens, _ = detect_sensitive_column(s_num, "MonthlyCharges")
    assert is_sens is False


# ── Dataset Context Extraction Tests ──────────────────────────────────────────


def test_extract_dataset_context():
    df = _sample_df()
    ctx = extract_dataset_context(
        df,
        target="Churn",
        user_description="Telco churn test dataset",
        excluded_columns={"email_address"},
    )
    assert ctx.row_count == 100
    assert ctx.column_count == 6
    assert ctx.target == "Churn"
    assert ctx.user_description == "Telco churn test dataset"
    assert "email_address" in ctx.excluded_columns

    # Check MonthlyCharges profile
    mc_profile = ctx.columns["MonthlyCharges"]
    assert mc_profile.profile_type == "numeric"
    assert mc_profile.min_val == pytest.approx(20.5)
    assert mc_profile.median_val is not None
    assert len(mc_profile.representative_samples) > 0

    # Excluded column should withhold samples
    email_profile = ctx.columns["email_address"]
    assert email_profile.is_excluded is True
    assert email_profile.representative_samples == []


# ── Cache Manager & Fingerprinting Tests ──────────────────────────────────────


def test_context_fingerprint_deterministic_and_sensitive():
    df = _sample_df()
    ctx1 = extract_dataset_context(df, target="Churn")
    ctx2 = extract_dataset_context(df, target="Churn")
    fp1 = compute_context_fingerprint(ctx1, "test-model")
    fp2 = compute_context_fingerprint(ctx2, "test-model")
    assert fp1 == fp2

    # Change in description changes fingerprint
    ctx_desc = extract_dataset_context(df, target="Churn", user_description="Changed description")
    fp_desc = compute_context_fingerprint(ctx_desc, "test-model")
    assert fp_desc != fp1

    # Change in target changes fingerprint
    ctx_no_target = extract_dataset_context(df, target=None)
    fp_no_target = compute_context_fingerprint(ctx_no_target, "test-model")
    assert fp_no_target != fp1


def test_cache_manager():
    cache = AICacheManager()
    df = _sample_df()
    ctx = extract_dataset_context(df)
    fp = compute_context_fingerprint(ctx, "mock-model")

    assert cache.get(fp) is None
    assert cache.is_stale(None, fp) is True

    dummy_result = AIUnderstandingResult(
        dataset_understanding=DatasetUnderstanding(
            description="Test dataset",
            possible_purpose="Testing",
            composition=["Test"],
            observations=[],
            confidence="high",
            limitations=[],
        ),
        columns={},
        metadata={"fingerprint": fp},
    )

    cache.put(fp, dummy_result)
    assert cache.get(fp) == dummy_result
    assert cache.is_stale(fp, fp) is False
    assert cache.is_stale(fp, "different-fp") is True


# ── Result Validator & Schema Tests ───────────────────────────────────────────


def test_validate_dataset_understanding_valid():
    raw = {
        "description": "Telco customer retention data",
        "possible_purpose": "Predict customer churn",
        "composition": ["Account info", "Billing"],
        "observations": [
            {
                "title": "Billing relation",
                "description": "Monthly and Total charges are correlated",
                "confidence": "high",
                "limitation": "Exact formula not given",
            }
        ],
        "confidence": "high",
        "limitations": ["Small sample size"],
    }
    du = validate_dataset_understanding(raw)
    assert du.description == "Telco customer retention data"
    assert du.confidence == "high"
    assert len(du.observations) == 1


def test_validate_dataset_understanding_malformed_fallback():
    raw = {"invalid_field": 123}
    du = validate_dataset_understanding(raw)
    assert du.confidence == "low"
    assert len(du.limitations) > 0


def test_validate_column_batch_with_missing_and_hallucinated_cols():
    raw_response = {
        "columns": [
            {
                "column_name": "MonthlyCharges",
                "likely_meaning": "Monthly bill",
                "semantic_type": "monetary",
                "secondary_semantic_types": ["continuous_numeric"],
                "expected_structure": {
                    "logical_type": "numerical",
                    "expected_format": "decimal",
                    "expected_constraints": ["non-negative"],
                    "expected_units": "USD",
                    "possible_range": ">= 0",
                },
                "observed_consistency": {
                    "status": "consistent",
                    "observations": ["Valid decimals"],
                },
                "confidence": "high",
                "limitations": [],
                "observations": [],
            },
            {
                "column_name": "HallucinatedColumn",
                "likely_meaning": "Should be ignored",
                "semantic_type": "unknown_ambiguous",
                "expected_structure": {"logical_type": "text"},
                "observed_consistency": {"status": "consistent"},
                "confidence": "low",
            },
        ]
    }

    # Expected batch includes MonthlyCharges and Contract (Contract was omitted in raw_response)
    batch_cols = ["MonthlyCharges", "Contract"]
    result = validate_column_batch(raw_response, batch_cols)

    assert "MonthlyCharges" in result
    assert result["MonthlyCharges"].semantic_type == "monetary"
    assert "HallucinatedColumn" not in result

    # Contract was omitted, should receive safe fallback entry
    assert "Contract" in result
    assert result["Contract"].observed_consistency.status == "insufficient_information"


# ── AI Client & Mock Tests ───────────────────────────────────────────────────


def test_mock_ai_client_dataset_and_columns():
    client = MockAIClient()
    df = _sample_df()
    ctx = extract_dataset_context(df, target="Churn")

    # Dataset prompt
    ds_res = client.generate_json("REQUIRED JSON SCHEMA: composition")
    assert "description" in ds_res
    assert "possible_purpose" in ds_res

    # Column prompt
    col_res = client.generate_json('COLUMNS TO INTERPRET: "column_name": "MonthlyCharges"')
    assert "columns" in col_res
    assert len(col_res["columns"]) > 0


def test_mock_ai_client_error_modes():
    rate_limit_client = MockAIClient(should_fail=True, failure_type="rate_limit")
    with pytest.raises(APIRateLimitError):
        rate_limit_client.generate_json("test")

    timeout_client = MockAIClient(should_fail=True, failure_type="timeout")
    with pytest.raises(APIConnectionError):
        timeout_client.generate_json("test")

    invalid_json_client = MockAIClient(should_fail=True, failure_type="invalid_json")
    with pytest.raises(InvalidJSONResponseError):
        invalid_json_client.generate_json("test")


def test_gemini_client_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class _EmptySecrets:
        def __contains__(self, item: str) -> bool:
            return False

    class _StreamlitWithoutSecrets:
        secrets = _EmptySecrets()

    monkeypatch.setitem(
        __import__("sys").modules,
        "streamlit",
        _StreamlitWithoutSecrets(),
    )
    with pytest.raises(APIKeyMissingError):
        GeminiAIClient(api_key=None)


def test_gemini_client_falls_back_when_configured_model_is_busy(monkeypatch):
    import json

    from data_doctor.ai_understanding import ai_client as ai_client_module

    calls: list[str] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self.text = json.dumps(payload)
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append(url)
        if "gemini-3.8-flash" in url:
            return FakeResponse(
                503,
                {"error": {"message": "high demand"}},
            )
        return FakeResponse(
            200,
            {
                "candidates": [
                    {"content": {"parts": [{"text": "{\"ok\": true}"}]}}
                ]
            },
        )

    monkeypatch.setattr(ai_client_module.requests, "post", fake_post)
    client = GeminiAIClient(
        api_key="test-key",
        model="gemini-3.8-flash",
        timeout_seconds=5,
    )

    assert client.generate_json("Reply with JSON") == {"ok": True}
    assert client.model == "gemini-flash-lite-latest"
    assert any("gemini-3.8-flash" in url for url in calls)
    assert any("gemini-flash-lite-latest" in url for url in calls)


# ── AIUnderstandingEngine End-to-End Tests ───────────────────────────────────


def test_engine_run_understanding_end_to_end():
    df = _sample_df()
    mock_client = MockAIClient()
    engine = AIUnderstandingEngine(client=mock_client)

    result = engine.run_understanding(
        df,
        target="Churn",
        user_description="Test telco dataset",
        excluded_columns={"email_address"},
    )

    assert isinstance(result, AIUnderstandingResult)
    assert result.dataset_understanding.description != ""
    assert "MonthlyCharges" in result.columns
    assert "Contract" in result.columns
    assert "customerID" in result.columns
    assert result.metadata["target"] == "Churn"
    assert "email_address" in result.metadata["excluded_columns"]

    # Verify cached return on second call
    call_count_before = mock_client.call_count
    cached_result = engine.run_understanding(
        df,
        target="Churn",
        user_description="Test telco dataset",
        excluded_columns={"email_address"},
    )
    assert cached_result.metadata["fingerprint"] == result.metadata["fingerprint"]
    assert mock_client.call_count == call_count_before


def test_ai_understanding_ui_integration(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from data_doctor.ai_understanding.ai_client import MockAIClient

    # Force MockAIClient during testing regardless of secrets.toml
    monkeypatch.setattr(
        "data_doctor.ai_understanding.AIUnderstandingEngine.get_client",
        lambda self: MockAIClient(),
    )

    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    assert not app.exception

    # Initially, AI understanding is NOT run automatically
    assert app.session_state.get("ai_understanding_result") is None
    assert any("AI Data Understanding" in exp.label for exp in app.expander)

    # Check consent checkbox
    consent_box = next(
        cb for cb in app.checkbox if "consent" in cb.key or "consent" in cb.label.lower()
    )
    consent_box.set_value(True).run(timeout=30)

    # Click Run AI Understanding
    run_btn = next(
        btn for btn in app.button if "Run AI Understanding" in btn.label
    )
    run_btn.click().run(timeout=30)
    assert not app.exception

    # Result should be stored in session state
    result = app.session_state.get("ai_understanding_result")
    assert result is not None
    assert isinstance(result, AIUnderstandingResult)
    assert not app.session_state.get("ai_is_stale", False)

    # Action log and working_df must remain completely untouched by AI analysis
    assert app.session_state["action_log"] == []

    # Applying a cleaning action should mark AI result as stale
    action_btn = next(
        btn for btn in app.button if btn.label == "Apply action"
    )
    action_btn.click().run(timeout=30)

    assert app.session_state.get("ai_is_stale") is True
    # Previous AI result remains preserved
    assert app.session_state.get("ai_understanding_result") is not None
