"""Action widgets for column cards (selectbox + apply button with feedback)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from streamlit_ui.actions import coerce_label_value, missing_like_label_values
from streamlit_ui.state import apply_working_action


def render_column_actions(
    column: str,
    series: pd.Series,
    applicable: dict[str, dict[str, Any]],
    key_prefix: str = "",
) -> None:
    """Render action selectbox + apply button with success/error feedback."""
    selected = st.selectbox(
        "Choose an action",
        list(applicable),
        format_func=lambda key: applicable[key]["label"],
        key=f"{key_prefix}column_action_{column}",
        label_visibility="collapsed",
    )
    option = applicable[selected]
    value: Any = None
    old_label: Any = None
    if option["action"] == "fill_constant":
        value = st.text_input(
            "Constant value",
            key=f"{key_prefix}constant_{column}",
        )
    elif option["action"] == "recode_label":
        old_label = st.selectbox(
            "Label to replace",
            missing_like_label_values(series),
            key=f"{key_prefix}recode_old_{column}",
        )
        value = st.text_input(
            "New label value",
            key=f"{key_prefix}recode_new_{column}",
        )
    disabled = (
        option["action"] in {"fill_constant", "recode_label"} and value == ""
    )
    action_value = value
    if option["action"] == "recode_label" and not disabled:
        try:
            action_value = {
                "old": old_label,
                "new": coerce_label_value(value, series),
            }
        except (ValueError, TypeError):
            action_value = value
    if st.button(
        "Apply action",
        key=f"{key_prefix}column_apply_{column}",
        disabled=disabled,
        width="stretch",
    ):
        try:
            if option["action"] == "recode_label":
                action_value = {
                    "old": old_label,
                    "new": coerce_label_value(value, series),
                }
            apply_working_action(
                option["action"],
                column=column,
                value=action_value,
            )
            action_label = applicable[selected]["label"]
            st.session_state["_last_action_success"] = (
                f"**{action_label}** applied to column **{column}**."
            )
            st.rerun()
        except (ValueError, TypeError) as error:
            st.session_state["_last_action_error"] = (
                f"Action failed on column **{column}**: {error}"
            )
            st.rerun()

