"""Session-state operations for the working dataset, including undo/redo."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_doctor.detectors import Finding, diagnose
from data_doctor.treatments import apply_action

MAX_UNDO_STEPS = 8


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
    st.session_state.undo_stack = []
    st.session_state.redo_stack = []
    st.session_state.column_page = 1
    st.session_state.expanded_column = None
    st.session_state.ai_understanding_result = None
    st.session_state.ai_context_fingerprint = None
    st.session_state.ai_is_stale = False
    st.session_state.ai_user_description = ""
    st.session_state.ai_excluded_columns = []
    st.session_state.ai_consent = False


def refresh_findings() -> list[Finding]:
    findings = diagnose_cached(
        st.session_state.working_df,
        st.session_state.target,
    )
    st.session_state.findings = findings
    return findings


def _snapshot() -> dict[str, Any]:
    """Capture the current working state for undo/redo."""
    return {
        "df": st.session_state.working_df.copy(deep=True),
        "action_log": list(st.session_state.action_log),
    }


def _restore_snapshot(snapshot: dict[str, Any]) -> None:
    """Restore working state from an undo/redo snapshot."""
    st.session_state.working_df = snapshot["df"]
    st.session_state.action_log = list(snapshot["action_log"])
    st.session_state.findings = None
    st.session_state.latest_training_result = None
    st.session_state.feature_drop_selection = []
    if st.session_state.get("ai_understanding_result") is not None:
        st.session_state.ai_is_stale = True


def push_undo() -> None:
    """Push the current working state onto the undo stack, clear redo."""
    stack: list = st.session_state.setdefault("undo_stack", [])
    stack.append(_snapshot())
    # Drop oldest entry when memory limit is exceeded
    if len(stack) > MAX_UNDO_STEPS:
        stack.pop(0)
    # Any new action invalidates the redo future
    st.session_state.setdefault("redo_stack", []).clear()


def undo_step() -> bool:
    """Restore the previous working state. Returns True if an undo was applied."""
    undo_stack: list = st.session_state.get("undo_stack", [])
    redo_stack: list = st.session_state.setdefault("redo_stack", [])
    if not undo_stack:
        return False
    # Push current state onto redo before restoring
    redo_stack.append(_snapshot())
    snapshot = undo_stack.pop()
    _restore_snapshot(snapshot)
    return True


def redo_step() -> bool:
    """Reapply the most recently undone state. Returns True if a redo was applied."""
    redo_stack: list = st.session_state.get("redo_stack", [])
    undo_stack: list = st.session_state.setdefault("undo_stack", [])
    if not redo_stack:
        return False
    # Push current state onto undo before restoring
    undo_stack.append(_snapshot())
    snapshot = redo_stack.pop()
    _restore_snapshot(snapshot)
    return True


def apply_working_action(
    action: str,
    column: str | None = None,
    value: Any = None,
) -> None:
    # Capture current state before mutation so undo can restore it
    push_undo()
    try:
        updated, log = apply_action(
            st.session_state.working_df,
            st.session_state.target,
            action,
            column=column,
            value=value,
        )
    except Exception:
        # If the action fails, pop the snapshot we just pushed so the
        # undo stack doesn't grow with a no-op entry
        undo_stack: list = st.session_state.get("undo_stack", [])
        if undo_stack:
            undo_stack.pop()
        raise
    st.session_state.working_df = updated
    st.session_state.action_log.append(log)
    if action != "keep_with_warning":
        st.session_state.findings = None
        st.session_state.latest_training_result = None
        if st.session_state.get("ai_understanding_result") is not None:
            st.session_state.ai_is_stale = True
    st.session_state.feature_drop_selection = []
