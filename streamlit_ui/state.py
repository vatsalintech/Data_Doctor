"""Session-state operations for the working dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding, diagnose
from data_doctor.treatments import apply_action


@st.cache_data(show_spinner=False)
def diagnose_cached(df: pd.DataFrame, target: str) -> list[Finding]:
    return diagnose(df, target)


def reset_for_source(
    df: pd.DataFrame,
    key: str,
    suggested_target: str | None = None,
) -> None:
    st.session_state.source_key = key
    st.session_state.original_df = df.copy(deep=True)
    st.session_state.working_df = df.copy(deep=True)
    st.session_state.target = suggested_target
    _reset_analysis_state()


def reset_for_target(target: str) -> None:
    st.session_state.working_df = st.session_state.original_df.copy(deep=True)
    st.session_state.target = target
    _reset_analysis_state()


def _reset_analysis_state() -> None:
    st.session_state.action_log = []
    st.session_state.run_history = []
    st.session_state.findings = None
    st.session_state.latest_training_result = None
    st.session_state.last_feature_relevance = {}
    st.session_state.show_relevance_scores = False
    st.session_state.feature_drop_selection = []


def refresh_findings() -> list[Finding]:
    findings = diagnose_cached(
        st.session_state.working_df,
        st.session_state.target,
    )
    st.session_state.findings = findings
    return findings


def apply_working_action(
    action: str,
    column: str | None = None,
    value: Any = None,
) -> None:
    updated, log = apply_action(
        st.session_state.working_df,
        st.session_state.target,
        action,
        column=column,
        value=value,
    )
    st.session_state.working_df = updated
    st.session_state.action_log.append(log)
    if action != "keep_with_warning":
        st.session_state.findings = None
        st.session_state.latest_training_result = None
    st.session_state.feature_drop_selection = []
