"""Dataset-wide issue cards."""

from typing import Any

import streamlit as st

from data_doctor.detectors import Finding
from streamlit_ui.components.common import render_severity
from streamlit_ui.state import apply_working_action


def dataset_issue_findings(findings: list[Finding]) -> list[Finding]:
    """Return issues important enough to keep above column pagination."""
    return [
        finding
        for finding in findings
        if finding.scope == "dataset"
        or (
            finding.issue_type == "stored_as_text"
            and finding.found
        )
    ]


def render_dataset_issues(
    findings: list[Finding],
    options: dict[str, dict[str, Any]],
) -> None:
    dataset_findings = dataset_issue_findings(findings)
    st.subheader("Dataset-wide issues")
    for start in range(0, len(dataset_findings), 3):
        columns = st.columns(3)
        for container, finding in zip(
            columns,
            dataset_findings[start : start + 3],
        ):
            with container:
                _render_dataset_issue_card(finding, options)


def _render_dataset_issue_card(
    finding: Finding,
    options: dict[str, dict[str, Any]],
) -> None:
    with st.container(border=True, height=588):
        title = finding.issue_type.replace("_", " ").title()
        if finding.issue_type == "stored_as_text":
            title = f"Stored as text: {finding.scope}"
        st.markdown(
            f'<div class="dd-issue-title">{title}</div>',
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
        }.get(finding.issue_type, finding.suggested_action)
        expected_column = (
            finding.scope
            if finding.issue_type == "stored_as_text"
            else None
        )
        applicable = {
            key: option
            for key, option in options.items()
            if option["column"] == expected_column
            and option["action"] == expected_action
        }
        if applicable and finding.found:
            _render_dataset_action(finding, applicable)
        elif (
            finding.issue_type == "class_imbalance"
            and finding.found
        ):
            st.caption(
                "Choose a class imbalance option in Controls, then train."
            )


def _render_dataset_action(
    finding: Finding,
    applicable: dict[str, dict[str, Any]],
) -> None:
    widget_suffix = (
        f"{finding.issue_type}_{finding.scope}".replace(" ", "_")
    )
    selected = st.selectbox(
        "Action",
        list(applicable),
        format_func=lambda key: applicable[key]["label"],
        key=f"dataset_action_{widget_suffix}",
    )
    if st.button(
        "Apply",
        key=f"dataset_apply_{widget_suffix}",
        use_container_width=True,
    ):
        try:
            option = applicable[selected]
            apply_working_action(
                option["action"],
                column=option["column"],
            )
            st.rerun()
        except (ValueError, TypeError) as error:
            st.error(str(error))
