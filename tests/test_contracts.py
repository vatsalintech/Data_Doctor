import pandas as pd

from data_doctor.contracts import (
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
