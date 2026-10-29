"""Top-level composition for the DataDoctor dashboard."""

import streamlit as st

from data_doctor.detectors import id_like_columns, prepare_data
from streamlit_ui.actions import action_options
from streamlit_ui.components.columns import render_column_grid
from streamlit_ui.components.common import render_preview, render_vitals
from streamlit_ui.components.controls import (
    render_control_panel,
    render_locked_target_panel,
    render_recovery_panel,
)
from streamlit_ui.components.dataset_issues import render_dataset_issues
from streamlit_ui.data_sources import render_source_selector
from streamlit_ui.state import (
    refresh_findings,
    reset_for_source,
)
from streamlit_ui.styles import configure_page, render_hero


def main() -> None:
    configure_page()
    render_hero()

    with st.container(border=True):
        loaded_df, source_key, suggested_target = render_source_selector()
    if loaded_df is None or source_key is None:
        st.info(
            "Choose a demo or upload a CSV to begin. Nothing is changed "
            "unless you explicitly apply an action."
        )
        return
    if st.session_state.get("source_key") != source_key:
        reset_for_source(loaded_df, source_key, suggested_target)

    working = st.session_state.working_df
    prepared = prepare_data(working)
    blocked_targets = set(id_like_columns(prepared))
    target_options = [
        column for column in prepared.columns if column not in blocked_targets
    ]
    if not target_options:
        st.error("No target candidate remains after excluding ID columns.")
        return
    if st.session_state.target not in target_options:
        st.session_state.target = None

    if st.session_state.target is None:
        _render_target_locked(working, target_options)
        return

    try:
        findings = (
            st.session_state.findings
            if st.session_state.findings is not None
            else refresh_findings()
        )
    except ValueError as error:
        _render_diagnosis_recovery(working, target_options, str(error))
        return

    options = action_options(
        findings,
        working,
        st.session_state.target,
    )
    main_column, control_column = st.columns([4.1, 1.4], gap="large")
    with control_column:
        render_control_panel(target_options, findings)
    with main_column:
        render_vitals(working)
        render_preview(working)
        render_dataset_issues(findings, options)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_column_grid(
        working,
        st.session_state.target,
        findings,
        options,
    )


def _render_target_locked(working, target_options: list[str]) -> None:
    main_column, control_column = st.columns([4.1, 1.4], gap="large")
    with control_column:
        render_locked_target_panel(target_options)
    with main_column:
        render_vitals(working)
        render_preview(working)
        st.info(
            "Choose the target column to unlock issue diagnosis, "
            "column analysis, and model training."
        )


def _render_diagnosis_recovery(
    working,
    target_options: list[str],
    error: str,
) -> None:
    main_column, control_column = st.columns([4.1, 1.4], gap="large")
    with control_column:
        render_recovery_panel(target_options, error)
    with main_column:
        render_vitals(working)
        render_preview(working)
        st.warning(
            "The working data cannot be diagnosed or trained in its "
            "current state. Change the target or reset to original."
        )
