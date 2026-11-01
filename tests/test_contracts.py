import numpy as np
import pandas as pd

from data_doctor.contracts import (
    _numeric_psi,
    classify_comparison,
    compare_datasets,
    infer_contract,
    validate_contract,
)


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": list(range(100)),
            "region": ["north", "south"] * 50,
            "label": ["yes", "no"] * 50,
        }
    )


def test_contract_flags_missing_and_unknown_values():
    reference = _reference()
    contract = infer_contract(reference, "label")
    current = reference.drop(columns=["age"]).copy()
    current.loc[0, "region"] = "west"
    violations = validate_contract(current, contract)
    checks = {(row["column"], row["check"]) for row in violations}
    assert ("age", "missing_column") in checks
    assert ("region", "unknown_values") in checks


def test_drift_flags_large_numeric_and_category_changes():
    reference = _reference()
    current = reference.copy()
    current["age"] = current["age"] + 1000
    current["region"] = "west"
    drift = compare_datasets(reference, current)
    checks = {(row["column"], row["check"]) for row in drift}
    assert ("age", "numeric_drift") in checks
    assert ("region", "category_drift") in checks


def test_expected_missingness_is_not_drift_and_schema_stays_separate():
    reference = _reference()
    contract = infer_contract(reference, "label")
    expected = reference.copy()
    expected.loc[:4, "age"] = np.nan
    assert validate_contract(expected, contract) == []
    assert compare_datasets(reference, expected) == []

    too_missing = reference.copy()
    too_missing.loc[:9, "age"] = np.nan
    violations = validate_contract(too_missing, contract)
    assert any(
        row["check"] == "missing_rate" and row["column"] == "age"
        for row in violations
    )
    assert all(row["check"] != "numeric_drift" for row in violations)

    dropped = reference.drop(columns=["age"])
    schema_issues = validate_contract(dropped, contract)
    schema_drift = compare_datasets(reference, dropped)
    sections = classify_comparison(schema_issues, schema_drift)
    assert any(row["check"] == "missing_column" for row in schema_issues)
    assert sections["distribution"] == []
    assert sections["unexpected_missingness"] == []


def test_high_missingness_can_still_break_the_ceiling():
    reference = _reference()
    reference["age"] = reference["age"].astype(float)
    reference.loc[:79, "age"] = np.nan
    contract = infer_contract(reference, "label")
    assert contract["columns"]["age"]["missing_rate_ceiling"] < 1
    current = reference.copy()
    current["age"] = np.nan
    violations = validate_contract(current, contract)
    drift = compare_datasets(reference, current)
    sections = classify_comparison(violations, drift)
    assert any(row["check"] == "missing_rate" for row in violations)
    assert sections["unexpected_missingness"] == []
    assert sections["distribution"] == []


def test_missingness_drop_is_not_called_distribution_drift():
    reference = _reference()
    reference.loc[0:8:2, "region"] = np.nan
    reference.loc[1:9:2, "region"] = np.nan
    current = _reference()
    contract = infer_contract(reference, "label")
    violations = validate_contract(current, contract)
    drift = compare_datasets(reference, current)
    sections = classify_comparison(violations, drift)
    assert not any(row["check"] == "missing_rate" for row in violations)
    assert sections["distribution"] == []
    assert [row["column"] for row in sections["unexpected_missingness"]] == [
        "region"
    ]


def test_psi_handles_collapsed_bins_and_empty_bins():
    reference = pd.DataFrame(
        {
            "flag": [0] * 95 + [1] * 5,
            "amount": np.linspace(0, 100, 100),
            "region": ["north"] * 100,
        }
    )
    shifted_flag = reference.copy()
    shifted_flag["flag"] = [0] * 5 + [1] * 95
    flag_drift = compare_datasets(reference, shifted_flag)
    assert ("flag", "numeric_drift") in {
        (row["column"], row["check"]) for row in flag_drift
    }

    same = _numeric_psi(reference["amount"], reference["amount"])
    assert same == 0
    moved = reference.copy()
    moved["amount"] = reference["amount"] + 10_000
    moved_psi = _numeric_psi(reference["amount"], moved["amount"])
    assert moved_psi is not None
    assert np.isfinite(moved_psi)
    assert moved_psi >= 0.20

    constant = pd.Series([5.0] * 40)
    relocated = _numeric_psi(constant, pd.Series([9.0] * 40))
    assert relocated is not None
    assert np.isfinite(relocated)
    assert relocated >= 0.20
