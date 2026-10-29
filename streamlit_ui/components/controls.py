"""Target, training, download, and recovery controls."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_doctor.contracts import (
    compare_datasets,
    contract_json,
    infer_contract,
    validate_contract,
)
from data_doctor.detectors import Finding
from data_doctor.report import (
    build_health_report,
    build_run_record,
    health_report_json,
)
from notebooks.model_training import train_model
from streamlit_ui.components.dialogs import (
    render_feature_relevance,
    render_training_results,
)
from streamlit_ui.data_sources import csv_bytes, read_uploaded_csv
from streamlit_ui.state import reset_for_target


def render_control_panel(
    target_options: list[str],
    findings: list[Finding],
) -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="control-title">Controls</div>',
            unsafe_allow_html=True,
        )
        current_target = st.session_state.target
        selected_target = st.selectbox(
            "Target column",
            target_options,
            index=(
                target_options.index(current_target)
                if current_target in target_options
                else 0
            ),
            help="ID-like columns are excluded.",
        )
        if selected_target != current_target:
            reset_for_target(selected_target)
            st.rerun()
        _render_training_controls(selected_target, findings)
        _render_working_data_controls(selected_target, findings)


def _render_training_controls(
    selected_target: str,
    findings: list[Finding],
) -> None:
    st.divider()
    st.markdown("**Model training**")
    st.caption(
        "For trustworthy metrics, row fixes affect training rows only. "
        "The working CSV applies your chosen cleaning to the full table."
    )
    open_issues = [finding for finding in findings if finding.found]
    high_issues = [
        finding for finding in open_issues if finding.severity == "high"
    ]
    with st.expander("Pre-train readiness"):
        st.info(
            "Checks passed so far. Final split and training checks run "
            "when you start training."
        )
        if high_issues:
            st.warning(
                f"{len(high_issues)} high-severity warning(s) remain. "
                "Training is allowed because warnings never auto-block "
                "a user choice."
            )
        else:
            st.success("No high-severity warnings remain.")
        st.caption(f"{len(open_issues)} total detected issue(s) remain open.")
    strategy = st.selectbox(
        "Class imbalance option",
        [
            "None",
            'class_weight="balanced"',
            "Random oversample",
            "Random undersample",
        ],
    )
    class_weight = (
        "balanced" if strategy == 'class_weight="balanced"' else None
    )
    sampling = {
        "Random oversample": "oversample",
        "Random undersample": "undersample",
    }.get(strategy)
    if st.button(
        "Train Logistic Regression",
        type="primary",
        key="train_model_button",
        use_container_width=True,
    ):
        _train(selected_target, findings, class_weight, sampling)

    result_available = st.session_state.latest_training_result is not None
    if st.button(
        "Training result",
        key="training_result_button",
        disabled=not result_available,
        use_container_width=True,
    ):
        render_training_results()
    if st.button(
        "Feature relevance",
        disabled=not result_available,
        use_container_width=True,
    ):
        render_feature_relevance()
    if not st.session_state.run_history:
        st.caption(
            "Train once to unlock the training result and feature relevance."
        )
    elif not result_available:
        st.caption(
            "Results are out of date after your latest data change. "
            "Train again to refresh them."
        )


def _train(
    selected_target: str,
    findings: list[Finding],
    class_weight: str | None,
    sampling: str | None,
) -> None:
    try:
        with st.spinner("Training…"):
            result = train_model(
                st.session_state.original_df,
                selected_target,
                class_weight=class_weight,
                sampling=sampling,
                actions=st.session_state.action_log,
            )
        record = build_run_record(
            result,
            st.session_state.working_df,
            findings,
            len(st.session_state.action_log),
        )
        record["run_number"] = len(st.session_state.run_history) + 1
        st.session_state.run_history.append(record)
        st.session_state.latest_training_result = result
        st.session_state.last_feature_relevance = dict(
            result["feature_relevance"]
        )
        st.rerun()
    except ValueError as error:
        st.error(str(error))


def _render_working_data_controls(
    selected_target: str,
    findings: list[Finding],
) -> None:
    st.divider()
    st.markdown("**Working data**")
    st.download_button(
        "Download working CSV",
        csv_bytes(st.session_state.working_df),
        file_name="datadoctor_working.csv",
        mime="text/csv",
        use_container_width=True,
    )
    report = build_health_report(
        st.session_state.working_df,
        selected_target,
        findings,
        st.session_state.action_log,
        st.session_state.run_history,
    )
    st.download_button(
        "Download data health report",
        health_report_json(report),
        file_name="datadoctor_health_report.json",
        mime="application/json",
        use_container_width=True,
    )
    contract = infer_contract(st.session_state.working_df, selected_target)
    st.download_button(
        "Download dataset contract",
        contract_json(contract),
        file_name="datadoctor_contract.json",
        mime="application/json",
        use_container_width=True,
    )
    _render_dataset_comparison(contract)
    if st.button("Reset to original", use_container_width=True):
        reset_for_target(selected_target)
        st.rerun()
    if st.session_state.action_log:
        with st.expander(
            f"Action log ({len(st.session_state.action_log)})"
        ):
            st.dataframe(
                pd.DataFrame(st.session_state.action_log),
                hide_index=True,
                width="stretch",
            )


def _render_dataset_comparison(contract: dict) -> None:
    with st.expander("Compare a new CSV"):
        comparison_upload = st.file_uploader(
            "New CSV",
            type=["csv"],
            key="comparison_upload",
        )
        if comparison_upload is None:
            return
        try:
            comparison_df = read_uploaded_csv(comparison_upload.getvalue())
            contract_issues = validate_contract(comparison_df, contract)
            drift = compare_datasets(
                st.session_state.working_df,
                comparison_df,
            )
            if not contract_issues and not drift:
                st.success(
                    "No contract violations or material drift signals "
                    "were found."
                )
            if contract_issues:
                st.markdown("**Contract violations**")
                st.dataframe(
                    pd.DataFrame(contract_issues),
                    hide_index=True,
                    width="stretch",
                )
            if drift:
                st.markdown("**Drift signals**")
                st.dataframe(
                    pd.DataFrame(drift),
                    hide_index=True,
                    width="stretch",
                )
        except (ValueError, OSError) as error:
            st.error(str(error))


def render_locked_target_panel(target_options: list[str]) -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="control-title">Controls</div>',
            unsafe_allow_html=True,
        )
        selected_target = st.selectbox(
            "Target column",
            target_options,
            index=None,
            placeholder="Choose the target column",
            help="ID-like columns are excluded.",
        )
        if selected_target is not None:
            reset_for_target(selected_target)
            st.rerun()
        st.caption("Choose a target to unlock diagnosis and training.")


def render_recovery_panel(
    target_options: list[str],
    error: str,
) -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="control-title">Recovery controls</div>',
            unsafe_allow_html=True,
        )
        current_target = st.session_state.target
        selected_target = st.selectbox(
            "Target column",
            target_options,
            index=target_options.index(current_target),
            key="recovery_target",
        )
        if selected_target != current_target:
            reset_for_target(selected_target)
            st.rerun()
        st.error(error)
        st.download_button(
            "Download working CSV",
            csv_bytes(st.session_state.working_df),
            file_name="datadoctor_working.csv",
            mime="text/csv",
            use_container_width=True,
            key="recovery_download",
        )
        if st.button(
            "Reset to original",
            use_container_width=True,
            key="recovery_reset",
        ):
            reset_for_target(selected_target)
            st.rerun()
