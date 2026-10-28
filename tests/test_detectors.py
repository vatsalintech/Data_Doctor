import json
from pathlib import Path

import pandas as pd

from data_doctor.detectors import (
    diagnose,
    id_like_columns,
    prepare_data,
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

