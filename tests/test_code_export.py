"""Unit tests for pipeline export and standalone cleaning code generation."""

import io
import pickle
import numpy as np
import pandas as pd
import pytest

try:
    import joblib
except ImportError:
    joblib = None

from data_doctor.code_export import export_model_bytes, generate_cleaning_code
from data_doctor.model_training import train_model


def test_export_model_bytes_with_pipeline():
    df = pd.DataFrame(
        {
            "num_feat": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] * 2,
            "cat_feat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 2,
            "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 2,
        }
    )
    result = train_model(df, "target", seed=42)
    pipeline = result["pipeline"]
    assert pipeline is not None

    data = export_model_bytes(pipeline)
    assert isinstance(data, bytes)
    assert len(data) > 0

    # Deserialization test
    if joblib is not None:
        restored = joblib.load(io.BytesIO(data))
    else:
        restored = pickle.loads(data)

    test_x = pd.DataFrame({"num_feat": [25.0, 75.0], "cat_feat": ["A", "B"]})
    orig_preds = pipeline.predict(test_x)
    rest_preds = restored.predict(test_x)
    assert np.array_equal(orig_preds, rest_preds)


def test_export_model_bytes_none_raises():
    with pytest.raises(ValueError, match="No fitted pipeline"):
        export_model_bytes(None)


def test_generate_cleaning_code_empty_log():
    code = generate_cleaning_code([], target="TargetCol")
    assert "def clean_dataset" in code
    assert "def normalize_missing_values" in code
    assert "No data cleaning actions were applied" in code
    # Valid python syntax
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None

    scope = {}
    exec(compiled, scope)
    assert "clean_dataset" in scope
    sample = pd.DataFrame({"TargetCol": [1, 2], "Other": ["a", "b"]})
    out = scope["clean_dataset"](sample)
    assert len(out) == 2


def test_generate_cleaning_code_full_execution():
    action_log = [
        {
            "action": "drop_exact_duplicates",
            "scope": "dataset",
        },
        {
            "action": "drop_column",
            "column": "unwanted_col",
            "scope": "unwanted_col",
        },
        {
            "action": "impute_median",
            "column": "num_col",
            "scope": "num_col",
            "details": {"fill_value": 50.0},
        },
        {
            "action": "cap_outliers",
            "column": "num_col",
            "scope": "num_col",
            "details": {"lower_bound": 10.0, "upper_bound": 90.0},
        },
        {
            "action": "group_rare_categories",
            "column": "cat_col",
            "scope": "cat_col",
            "details": {"grouped_values": ["rare_val"]},
        },
        {
            "action": "recode_label",
            "column": "target",
            "scope": "target",
            "details": {"old": "Y", "new": "Yes"},
        },
        {
            "action": "keep_with_warning",
            "column": "cat_col",
            "scope": "cat_col",
            "details": {"warning_acknowledged": True},
        },
    ]

    code = generate_cleaning_code(action_log, target="target", source_name="test.csv")
    assert "def clean_dataset" in code
    assert "Step 1: Drop exact duplicate rows" in code
    assert "Step 2: Drop column 'unwanted_col'" in code
    assert "Step 3: Impute missing values in 'num_col' with median (50.0)" in code
    assert "Step 4: Cap outliers in 'num_col' to range [10.0, 90.0]" in code
    assert "Step 5: Group rare categories in 'cat_col' into 'Other'" in code
    assert "Step 6: Recode target label in 'target' from 'Y' to 'Yes'" in code

    # Execute code in isolated namespace
    compiled = compile(code, "<string>", "exec")
    scope = {}
    exec(compiled, scope)
    clean_func = scope["clean_dataset"]

    raw_df = pd.DataFrame(
        {
            "unwanted_col": [1, 2, 3, 1],
            "num_col": [5.0, np.nan, 150.0, 5.0],
            "cat_col": ["common", "rare_val", "common", "common"],
            "target": ["Y", "No", "Yes", "Y"],
        }
    )

    cleaned = clean_func(raw_df)

    # 1. Duplicates dropped (row 0 and row 3 are duplicates)
    assert len(cleaned) == 3

    # 2. Dropped column removed
    assert "unwanted_col" not in cleaned.columns

    # 3. Missing imputed
    assert cleaned["num_col"].isna().sum() == 0

    # 4. Outliers capped: 5.0 -> 10.0, 150.0 -> 90.0
    assert cleaned["num_col"].min() >= 10.0
    assert cleaned["num_col"].max() <= 90.0

    # 5. Rare category grouped
    assert "rare_val" not in cleaned["cat_col"].values
    assert "Other" in cleaned["cat_col"].values

    # 6. Recoded label: "Y" became "Yes"
    assert "Y" not in cleaned["target"].values
    assert "Yes" in cleaned["target"].values
