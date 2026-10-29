"""Small visual components shared by dashboard sections."""

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding, dataset_vitals, prepare_data
from streamlit_ui.config import SEVERITY_ORDER


def render_vitals(df: pd.DataFrame) -> None:
    vitals = dataset_vitals(prepare_data(df))
    columns = st.columns(4)
    columns[0].metric("Rows", f"{vitals['rows']:,}")
    columns[1].metric("Columns", f"{vitals['columns']:,}")
    columns[2].metric(
        "Missing cells",
        f"{vitals['missing_cells']:,}",
        help=f"{vitals['missing_percent']:.2f}% of all cells",
    )
    columns[3].metric(
        "Duplicate extras",
        f"{vitals['exact_duplicate_extras']:,}",
    )


def render_preview(df: pd.DataFrame) -> None:
    with st.expander("Preview data"):
        st.dataframe(df.head(10), width="stretch")


def render_severity(severity: str) -> None:
    label = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "none": "No issue found",
    }.get(severity, severity.title())
    safe_severity = (
        severity if severity in {"high", "medium", "low"} else "none"
    )
    st.markdown(
        f'<span class="dd-severity dd-severity-{safe_severity}">'
        f"{label}</span>",
        unsafe_allow_html=True,
    )


def highest_severity(findings: list[Finding]) -> str:
    if not findings:
        return "none"
    return max(
        (finding.severity for finding in findings),
        key=lambda value: SEVERITY_ORDER.get(value, 0),
    )
