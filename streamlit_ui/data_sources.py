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
        demo_name = st.selectbox("Demo", list(DEMO_FILES))
        path = DEMO_FILES[demo_name]
        return (
            read_demo(str(path)),
            source_key(str(path)),
            DEMO_TARGETS[demo_name],
        )

    upload = st.file_uploader("Upload a CSV", type=["csv"])
    st.caption("Large files can take a few minutes to train.")
    if upload is None:
        return None, None, None
    content = upload.getvalue()
    try:
        return (
            read_uploaded_csv(content),
            source_key(upload.name, content),
            None,
        )
    except (ValueError, OSError) as error:
        st.error(f"Could not read this CSV: {error}")
        return None, None, None
