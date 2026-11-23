"""Export utilities for trained models and standalone data cleaning code."""

from __future__ import annotations

import io
import pickle
from datetime import datetime, timezone
from typing import Any

try:
    import joblib  # type: ignore
except ImportError:
    joblib = None


def export_model_bytes(pipeline: Any) -> bytes:
    """Serialize a fitted scikit-learn Pipeline into downloadable bytes (.joblib / .pkl)."""
    if pipeline is None:
        raise ValueError("No fitted pipeline was provided.")

    if joblib is not None:
        buf = io.BytesIO()
        joblib.dump(pipeline, buf)
        return buf.getvalue()

    return pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)


def generate_cleaning_code(
    action_log: list[dict[str, Any]],
    target: str,
    source_name: str = "dataset.csv",
) -> str:
    """Generate standalone, reproducible Python code replicating the applied cleaning steps.

    The generated script takes raw input data, normalizes missing/sentinel values,
    and applies every user action in exact chronological sequence.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        '"""DataDoctor Data Cleaning Pipeline.',
        f"Generated automatically on: {timestamp}",
        f"Target column: {target}",
        f"Source reference: {source_name}",
        f"Total cleaning steps applied: {len(action_log)}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "import numpy as np",
        "import pandas as pd",
        "",
        "",
        "def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:",
        '    """Convert standard text missing stand-ins and numeric sentinels into np.nan."""',
        "    cleaned = df.copy()",
        "    text_missing = {",
        '        "", "?", "??", "???", "*", "-", "--", ".", "na", "n/a", "n.a",',
        '        "n.a.", "na.", "#n/a", "#na", "<na>", "none", "null", "nil", "nan",',
        '        "nat", "unknown", "missing", "not available", "not applicable",',
        '        "unavailable", "unspecified", "not known", "not specified",',
        '        "#value!", "#div/0!", "#ref!", "#name?", "#null!", "#num!",',
        "    }",
        "    sentinel_numbers = {-99, -999, -9999}",
        "    for col in cleaned.columns:",
        "        series = cleaned[col]",
        "        if pd.api.types.is_string_dtype(series) or series.dtype == object:",
        "            key = series.astype(str).str.strip().str.lower()",
        "            cleaned.loc[key.isin(text_missing), col] = np.nan",
        "        elif pd.api.types.is_numeric_dtype(series):",
        "            cleaned.loc[series.isin(sentinel_numbers), col] = np.nan",
        "    return cleaned",
        "",
        "",
        f"def clean_dataset(df: pd.DataFrame, target: str = {repr(target)}) -> pd.DataFrame:",
        '    """Execute all DataDoctor cleaning actions on the input DataFrame in order."""',
        "    df = df.copy()",
        "    df = normalize_missing_values(df)",
        "",
    ]

    if not action_log:
        lines.append("    # No data cleaning actions were applied; returning normalized data.")
        lines.append("    return df")
    else:
        for index, entry in enumerate(action_log, start=1):
            action = entry.get("action", "")
            column = entry.get("column")
            details = entry.get("details", {})

            col_repr = repr(column) if column else "None"
            target_repr = repr(target)

            if action == "drop_exact_duplicates":
                lines.append(f"    # Step {index}: Drop exact duplicate rows")
                lines.append("    df = df.drop_duplicates().reset_index(drop=True)")
            elif action == "drop_conflicting_rows":
                lines.append(f"    # Step {index}: Drop rows with identical features but conflicting {target_repr}")
                lines.append(f"    features = [col for col in df.columns if col != {target_repr}]")
                lines.append("    if features:")
                lines.append(f"        counts = df.groupby(features, dropna=False)[{target_repr}].transform('nunique')")
                lines.append("        df = df.loc[counts <= 1].reset_index(drop=True)")
            elif action == "drop_rows_missing":
                lines.append(f"    # Step {index}: Drop rows where column {col_repr} is missing")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        df = df.dropna(subset=[{col_repr}]).reset_index(drop=True)")
            elif action == "drop_weird_label_rows":
                lines.append(f"    # Step {index}: Drop rows with missing or unparseable target values in {target_repr}")
                lines.append(f"    if {target_repr} in df.columns:")
                lines.append(f"        df = df.dropna(subset=[{target_repr}]).reset_index(drop=True)")
            elif action == "drop_column":
                lines.append(f"    # Step {index}: Drop column {col_repr}")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        df = df.drop(columns=[{col_repr}])")
            elif action in {"impute_mean", "impute_median", "impute_mode"}:
                method = action.replace("impute_", "")
                fill_val = details.get("fill_value")
                lines.append(f"    # Step {index}: Impute missing values in {col_repr} with {method} ({repr(fill_val)})")
                lines.append(f"    if {col_repr} in df.columns:")
                if fill_val is not None:
                    lines.append(f"        df[{col_repr}] = df[{col_repr}].fillna({repr(fill_val)})")
                else:
                    if method == "mean":
                        lines.append(f"        df[{col_repr}] = df[{col_repr}].fillna(df[{col_repr}].mean())")
                    elif method == "median":
                        lines.append(f"        df[{col_repr}] = df[{col_repr}].fillna(df[{col_repr}].median())")
                    else:
                        lines.append(f"        df[{col_repr}] = df[{col_repr}].fillna(df[{col_repr}].mode().iloc[0])")
            elif action == "fill_constant":
                fill_val = details.get("fill_value")
                lines.append(f"    # Step {index}: Fill missing values in {col_repr} with constant {repr(fill_val)}")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        df[{col_repr}] = df[{col_repr}].fillna({repr(fill_val)})")
            elif action == "convert_to_number":
                lines.append(f"    # Step {index}: Convert column {col_repr} from text to numeric")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        df[{col_repr}] = pd.to_numeric(df[{col_repr}], errors='coerce')")
            elif action == "convert_to_datetime":
                lines.append(f"    # Step {index}: Convert column {col_repr} to datetime")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        df[{col_repr}] = pd.to_datetime(df[{col_repr}], errors='coerce', format='mixed')")
            elif action == "cap_outliers":
                low = details.get("lower_bound")
                high = details.get("upper_bound")
                lines.append(f"    # Step {index}: Cap outliers in {col_repr} to range [{low}, {high}]")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        df[{col_repr}] = df[{col_repr}].clip(lower={repr(low)}, upper={repr(high)})")
            elif action == "remove_outliers":
                low = details.get("lower_bound")
                high = details.get("upper_bound")
                lines.append(f"    # Step {index}: Remove rows where {col_repr} is outside [{low}, {high}]")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        mask = df[{col_repr}].between({repr(low)}, {repr(high)}) | df[{col_repr}].isna()")
                lines.append("        df = df.loc[mask].reset_index(drop=True)")
            elif action == "group_rare_categories":
                rare_values = details.get("grouped_values", [])
                lines.append(f"    # Step {index}: Group rare categories in {col_repr} into 'Other'")
                lines.append(f"    if {col_repr} in df.columns:")
                lines.append(f"        rare_vals = set({repr(rare_values)})")
                lines.append(f"        df[{col_repr}] = df[{col_repr}].apply(lambda v: 'Other' if str(v) in rare_vals else v)")
            elif action == "recode_label":
                old_val = details.get("old")
                new_val = details.get("new")
                lines.append(f"    # Step {index}: Recode target label in {target_repr} from {repr(old_val)} to {repr(new_val)}")
                lines.append(f"    if {target_repr} in df.columns:")
                lines.append(f"        df[{target_repr}] = df[{target_repr}].replace({{{repr(old_val)}: {repr(new_val)}}})")
            elif action == "keep_with_warning":
                lines.append(f"    # Step {index}: Acknowledged review warning for {col_repr} (no data change)")
            else:
                lines.append(f"    # Step {index}: Unknown or custom action {repr(action)} on {col_repr}")

            lines.append("")

        lines.append("    return df")

    lines.extend([
        "",
        "",
        'if __name__ == "__main__":',
        f'    input_file = sys.argv[1] if len(sys.argv) > 1 else {repr(source_name)}',
        '    output_file = sys.argv[2] if len(sys.argv) > 2 else "cleaned_" + input_file',
        '    print(f"Loading {input_file}...")',
        "    try:",
        "        raw_data = pd.read_csv(input_file)",
        '        print(f"Original shape: {raw_data.shape}")',
        "        cleaned_data = clean_dataset(raw_data)",
        '        print(f"Cleaned shape:  {cleaned_data.shape}")',
        "        cleaned_data.to_csv(output_file, index=False)",
        '        print(f"Successfully saved cleaned data to {output_file}")',
        "    except FileNotFoundError:",
        '        print(f"Error: {input_file} not found. Please provide a valid CSV path as an argument.")',
        "        sys.exit(1)",
        "",
    ])

    return "\n".join(lines)
