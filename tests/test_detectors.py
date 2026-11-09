import json
from pathlib import Path

import pandas as pd

from data_doctor.detectors import (
    diagnose,
    id_like_columns,
    prepare_data,
    stored_as_text_findings,
)
from data_doctor.treatments import apply_action

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


def _found(df: pd.DataFrame, target: str, issue: str, column: str) -> bool:
    return any(
        finding.found
        and finding.issue_type == issue
        and finding.scope == column
        for finding in diagnose(df, target)
    )


def test_real_world_missing_standins_are_cleared_by_mode_fill():
    glucose = (
        ["75"] * 10
        + ["81", "82"]
        + ["?", "unknown", "missing", "   ", None, "-999", "N/A"]
        + [" 80 "]
    )
    labels = ["yes", "no"] * 10
    df = pd.DataFrame({"glucose": glucose, "label": labels})
    assert _found(df, "label", "missing_values", "glucose")

    updated, log = apply_action(df, "label", "impute_mode", column="glucose")

    assert log["details"]["fill_value"] == "75"
    assert not _found(updated, "label", "missing_values", "glucose")
    assert _found(updated, "label", "stored_as_text", "glucose")
    assert "unknown" not in set(updated["glucose"].astype(str))
    assert "-999" not in set(updated["glucose"].astype(str))


def test_spelling_variants_are_flagged_and_left_unchanged():
    sex = (
        ["female", "Female", "FEMALE", "f", "0", " 0 ", "male"] * 2
        + ["female", "male", "woman", "F", "m", "1"]
    )
    labels = ["yes", "no"] * 10
    df = pd.DataFrame({"Sex": sex, "label": labels})

    updated, _ = apply_action(df, "label", "impute_mode", column="Sex")

    assert updated["Sex"].tolist() == df["Sex"].tolist()
    review = next(
        finding
        for finding in diagnose(updated, "label")
        if finding.issue_type == "manual_review" and finding.scope == "Sex"
    )
    assert review.found
    assert review.severity == "low"
    assert "will not guess" in review.reason
    assert "female" in updated["Sex"].tolist()
    assert "Female" in updated["Sex"].tolist()


def test_many_target_spellings_raise_a_manual_alert():
    labels = (
        ["No", "no", "NO", "false", "0", "N"] * 2
        + ["Yes", "yes", "TRUE", "1", "Y"] * 2
    )
    df = pd.DataFrame(
        {"feature": list(range(len(labels))), "TenYearCHD": labels}
    )
    review = next(
        finding
        for finding in diagnose(df, "TenYearCHD")
        if finding.issue_type == "manual_review"
        and finding.scope == "TenYearCHD"
    )
    assert review.severity == "high"
    assert "will not guess" in review.reason
    assert df["TenYearCHD"].nunique() == 11


def test_blank_column_names_are_reported_without_being_renamed():
    df = pd.DataFrame(
        {
            " Sex ": ["female", "male"] * 10,
            "label": ["yes", "no"] * 10,
        }
    )
    review = next(
        finding
        for finding in diagnose(df, "label")
        if finding.issue_type == "manual_review" and finding.scope == "dataset"
    )
    assert " Sex " in review.reason
    assert list(df.columns) == [" Sex ", "label"]


def test_unreadable_numbers_are_flagged_and_not_converted():
    age = ["40", "41", "42", "43", "44", "45", "46 yrs", "fifty", "44yo"]
    age = age + ["40", "41", "42", "43", "44", "45", "47", "48", "49", "50", "51"]
    df = pd.DataFrame({"age": age, "label": ["yes", "no"] * 10})
    review = next(
        finding
        for finding in diagnose(df, "label")
        if finding.issue_type == "manual_review" and finding.scope == "age"
    )
    assert "46 yrs" in review.reason
    assert "will not guess" in review.reason
    assert "46 yrs" in set(df["age"])


def test_numeric_placeholder_is_reported_and_not_imputed():
    heart = pd.Series([-1] + list(range(60, 99)), dtype="float")
    df = pd.DataFrame(
        {"heart_rate": heart, "label": ["yes", "no"] * 20}
    )
    assert not _found(df, "label", "missing_values", "heart_rate")
    review = next(
        finding
        for finding in diagnose(df, "label")
        if finding.issue_type == "manual_review"
        and finding.scope == "heart_rate"
    )
    assert "-1" in review.reason
    assert "will not treat them as missing" in review.reason

