"""Dataset-wide issue cards."""

from typing import Any

import streamlit as st

from data_doctor.detectors import Finding
from streamlit_ui.components.common import render_severity
from streamlit_ui.state import apply_working_action


def render_dataset_issues(
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    dataset_findings = [
        finding for finding in findings if finding.scope == "dataset"
    ]
    st.subheader("Dataset-wide issues")
    columns = st.columns(max(1, len(dataset_findings)))
    for container, finding in zip(columns, dataset_findings):
        with container:
            with st.container(border=True, height=588):
                st.markdown(
                    '<div class="dd-issue-title">'
                    f"{finding.issue_type.replace('_', ' ').title()}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="dd-section-label">Severity</div>',
                    unsafe_allow_html=True,
                )
                render_severity(finding.severity)
                st.caption(finding.reason)
                if finding.found:
                    evidence_text = " · ".join(
                        f"{key.replace('_', ' ').title()}: {value}"
                        for key, value in finding.evidence.items()
                    )
                    st.caption(evidence_text)
                expected_action = {
                    "exact_duplicates": "drop_exact_duplicates",
                    "conflicting_labels": "drop_conflicting_rows",
                }.get(finding.issue_type)
                applicable = {
                    key: option
                    for key, option in options.items()
                    if option["column"] is None
                    and option["action"] == expected_action
                }
                if applicable and finding.found:
                    _render_dataset_action(finding, applicable)
                elif (
                    finding.issue_type == "class_imbalance"
                    and finding.found
                ):
                    st.caption(
                        "Choose a class imbalance option in Controls, "
                        "then train."
                    )


def _render_dataset_action(
    finding: Finding,
    applicable: dict[str, dict[str, Any]],
) -> None:
    selected = st.selectbox(
        "Action",
        list(applicable),
        format_func=lambda key: applicable[key]["label"],
        key=f"dataset_action_{finding.issue_type}",
    )
    if st.button(
        "Apply",
        key=f"dataset_apply_{finding.issue_type}",
        use_container_width=True,
    ):
        try:
            apply_working_action(applicable[selected]["action"])
            st.rerun()
        except (ValueError, TypeError) as error:
            st.error(str(error))
