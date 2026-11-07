"""CSV loading, validation, and source selection."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st

from streamlit_ui.config import (
    DEMO_FILES,
    DEMO_TARGETS,
    MAX_COLUMNS,
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
)

# Heuristic thresholds for dataset type detection
_IMAGE_COL_NAMES = {"pixel", "px", "r_", "g_", "b_", "red", "green", "blue"}
_MIN_PIXEL_COLS = 50  # > 50 numeric columns named pixel-like → likely image flatten


def _detect_non_tabular_issues(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable warnings if the dataset looks non-tabular."""
    issues: list[str] = []

    # 1. Single column (likely not a structured table)
    if len(df.columns) == 1:
        issues.append(
            "The file has only one column. "
            "Classification datasets need at least a target column and one feature column."
        )

    # 2. Flattened image dataset heuristic: many columns that look like pixel values
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    pixel_like = [
        c for c in numeric_cols
        if any(keyword in c.lower() for keyword in _IMAGE_COL_NAMES)
        or c.lower().startswith("pixel")
    ]
    if len(pixel_like) >= _MIN_PIXEL_COLS:
        issues.append(
            f"This looks like a flattened image dataset ({len(pixel_like)} pixel-named columns). "
            "DataDoctor is designed for structured tabular data only — it does not support "
            "image, audio, or text datasets."
        )

    # 3. All columns are numeric and there are hundreds of them (another image/signal hint)
    if len(numeric_cols) == len(df.columns) and len(df.columns) > 200:
        issues.append(
            f"The file has {len(df.columns)} entirely numeric columns with no text features. "
            "This may be a flattened image, audio, or signal dataset. "
            "DataDoctor works best with structured tabular data that has a mix of "
            "numeric and categorical columns."
        )

    # 4. Columns that look like free-text / long NLP paragraphs
    long_text_cols = []
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(50)
            if len(sample) > 0:
                avg_chars = sample.astype(str).str.len().mean()
                avg_words = sample.astype(str).str.split().str.len().mean()
                if avg_chars > 150 and avg_words > 20:
                    long_text_cols.append(col)
    if long_text_cols:
        col_list = ", ".join(f"'{c}'" for c in long_text_cols[:3])
        more = f" and {len(long_text_cols) - 3} more" if len(long_text_cols) > 3 else ""
        issues.append(
            f"Column(s) {col_list}{more} appear to contain long free-text sentences or paragraphs. "
            "DataDoctor is designed for structured classification data, not NLP/text datasets. "
            "If these columns are raw text descriptions, consider excluding them from modeling."
        )

    return issues


@st.cache_data(show_spinner=False)
def read_demo(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"CSV files must be {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
            "or smaller."
        )
    try:
        frame = pd.read_csv(BytesIO(content))
    except UnicodeDecodeError:
        try:
            frame = pd.read_csv(BytesIO(content), encoding="latin-1")
        except Exception as error:
            raise ValueError(
                "The file encoding could not be read as UTF-8 or Latin-1."
            ) from error
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The CSV has inconsistent rows or delimiters."
        ) from error
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The CSV has no usable rows or columns.")
    if len(frame.columns) == 1 and any(
        delimiter in str(frame.columns[0]) for delimiter in (";", "\t", "|")
    ):
        raise ValueError(
            "This file appears to use a non-comma delimiter. Export it as "
            "a comma-separated CSV and try again."
        )
    if len(frame) < 10:
        raise ValueError(
            f"The file has only {len(frame)} row(s). "
            "DataDoctor requires at least 10 rows for train/test splitting."
        )
    if len(frame) > MAX_ROWS:
        raise ValueError(f"CSV files may contain at most {MAX_ROWS:,} rows.")
    if len(frame.columns) > MAX_COLUMNS:
        raise ValueError(
            f"CSV files may contain at most {MAX_COLUMNS:,} columns."
        )
    return frame


@st.cache_data(show_spinner=False)
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def source_key(name: str, content: bytes | None = None) -> str:
    digest = hashlib.sha256(content).hexdigest() if content else name
    return f"{name}:{digest}"


def render_source_selector(
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Render the dataset selector and return its data and default target."""
    source_mode = st.radio(
        "Data source",
        ["Demo dataset", "Upload CSV"],
        horizontal=True,
    )
    if source_mode == "Demo dataset":
        available_demos = {
            name: path for name, path in DEMO_FILES.items() if path.exists()
        }
        if not available_demos:
            st.warning("No demo datasets found. Please upload a CSV.")
            return None, None, None
        demo_name = st.selectbox("Demo", list(available_demos))
        path = available_demos[demo_name]
        return (
            read_demo(str(path)),
            source_key(str(path)),
            DEMO_TARGETS[demo_name],
        )

    upload = st.file_uploader("Upload a CSV", type=["csv"])
    st.caption(
        "Large files can take a few minutes to train. On free cloud hosting "
        "(~1 GB RAM), CSVs under 20,000 rows (≤ 10 MB) are recommended."
    )
    if upload is None:
        return None, None, None
    content = upload.getvalue()
    try:
        df = read_uploaded_csv(content)
    except (ValueError, OSError) as error:
        st.error(f"Could not read this CSV: {error}")
        return None, None, None

    # Run non-tabular / unsupported dataset checks
    issues = _detect_non_tabular_issues(df)
    if issues:
        with st.container():
            st.warning(
                "⚠️ **This dataset may not be suitable for DataDoctor.**\n\n"
                "DataDoctor is designed exclusively for **structured tabular "
                "classification data** — think customer churn, credit risk, or "
                "medical diagnosis tables where each row is an observation with "
                "a mix of numeric and categorical features and one target column.\n\n"
                "**What we detected:**\n"
                + "\n".join(f"- {issue}" for issue in issues)
                + "\n\n"
                "**Supported:** Structured CSV with numeric/categorical columns "
                "and a single classification target (2–20 classes).\n\n"
                "**Not supported:** Image datasets, NLP/text corpora, audio "
                "feature matrices, time-series sequences, or regression targets.",
                icon="⚠️",
            )
        # Still return the dataframe — user may want to proceed anyway
    return (
        df,
        source_key(upload.name, content),
        None,
    )

