import json
from pathlib import Path

import pandas as pd

from data_doctor.detectors import (
    diagnose,
    id_like_columns,
    prepare_data,
    stored_as_text_findings,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generated_datasets_match_manifest():
    manifest = json.loads(
        (ROOT / "data" / "Messy" / "issue_manifest.json").read_text()
    )
    for filename, metadata in manifest["datasets"].items():
        df = pd.read_csv(ROOT / "data" / "Messy" / filename)
        observed = metadata["observed"]
        assert df.shape == (observed["rows"], observed["columns"])
        assert int(df.duplicated().sum()) == observed[
            "exact_duplicate_extras"
        ]
        assert metadata["target"] in df.columns
        assert metadata["target_copy"] in df.columns
        assert metadata["id_column"] in df.columns


def test_detectors_use_any_target_name():
    df = pd.DataFrame(
        {
            "row_key": [f"id-{i}" for i in range(30)],
            "number": list(range(29)) + [9999],
            "category": ["a", "b"] * 15,
            "answer": ["yes", "no"] * 15,
            "answer_copy": ["yes", "no"] * 15,
        }
    )
    findings = diagnose(df, "answer")
    found_types = {
        (finding.issue_type, finding.scope)
        for finding in findings
        if finding.found
    }
    assert "row_key" in id_like_columns(df)
    assert ("target_copy", "answer_copy") in found_types
    assert any(
        finding.issue_type == "outliers"
        and finding.scope == "number"
        and finding.found
        for finding in findings
    )


def test_prepare_data_recognizes_blank_and_question_mark():
    df = pd.DataFrame({"feature": ["ok", " ", "?"], "label": [0, 1, 0]})
    prepared = prepare_data(df)
    assert prepared["feature"].isna().sum() == 2


def test_stored_as_text_detects_numbers_and_dates_but_not_padded_ids():
    df = pd.DataFrame(
        {
            "amount": ["1.5", "2.0", "3.25", None],
            "event_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                None,
            ],
            "postal_code": ["00101", "00102", "00103", "00104"],
            "non_finite": ["1.0", "2.0", "3.0", "inf"],
            "label": ["yes", "no", "yes", "no"],
        }
    )
    findings = stored_as_text_findings(prepare_data(df), "label")
    inferred = {
        finding.scope: finding.evidence["inferred_type"]
        for finding in findings
    }
    assert inferred == {"amount": "number", "event_date": "date"}


def test_clean_telco_total_charges_is_detected_as_numeric_text():
    df = pd.read_csv(ROOT / "data" / "Original" / "Telco-Customer-Churn.csv")
    findings = stored_as_text_findings(prepare_data(df), "Churn")
    assert any(
        finding.scope == "TotalCharges"
        and finding.evidence["inferred_type"] == "number"
        for finding in findings
    )


def test_datetime_columns_are_not_treated_as_ids():
    rows = 30
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=rows),
            "feature": [index % 4 for index in range(rows)],
            "label": ["yes", "no"] * (rows // 2),
        }
    )
    assert "event_date" not in id_like_columns(df)
    findings = diagnose(df, "label")
    assert not any(
        finding.issue_type == "id_like"
        and finding.scope == "event_date"
        for finding in findings
    )


def test_leakage_detector_flags_near_perfect_target_mapping():
    labels = ["class-a", "class-b", "class-c"] * 40
    mapped = {
        "class-a": "north",
        "class-b": "south",
        "class-c": "west",
    }
    df = pd.DataFrame(
        {
            "region_code": [mapped[label] for label in labels],
            "noise": list(range(len(labels))),
            "outcome": labels,
        }
    )

    findings = diagnose(df, "outcome")

    assert any(
        finding.issue_type == "suspicious_predictor"
        and finding.scope == "region_code"
        and finding.found
        for finding in findings
    )


def test_leakage_detector_ignores_high_cardinality_noise():
    rows = 100
    df = pd.DataFrame(
        {
            "code": [f"group-{index // 2}" for index in range(rows)],
            "feature": [index % 7 for index in range(rows)],
            "outcome": ["yes", "no"] * (rows // 2),
        }
    )
    findings = diagnose(df, "outcome")
    assert not any(
        finding.issue_type == "suspicious_predictor"
        and finding.scope == "code"
        for finding in findings
    )

