"""Target, training, download, and recovery controls."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_doctor.contracts import (
    classify_comparison,
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
from streamlit_ui.state import redo_step, reset_for_target, undo_step


def render_control_panel(
    target_options: list[str],
    findings: list[Finding],
) -> None:
    with st.container(border=True, key="dashboard_controls_panel"):
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
    open_issues = [finding for finding in findings if finding.found]
    high_issues = [
        finding for finding in open_issues if finding.severity == "high"
    ]
    warning_text = (
        f"{len(high_issues)} high-severity warning(s) remain. Training is "
        "allowed because warnings never auto-block a user choice."
        if high_issues
        else "No high-severity warnings remain."
    )
    training_help = (
        "For trustworthy metrics, row fixes affect training rows only. "
        "The working CSV applies your chosen cleaning to the full table.\n\n"
        "**Pre-train readiness:**\n\n"
        "- Checks passed so far. Final split and training checks run when you start training.\n"
        f"- {warning_text}\n"
        f"- {len(open_issues)} total detected issue(s) remain open."
    )
    st.markdown("**Model training**", help=training_help)

    # Imbalance and positive class selectors (stacked one below another)
    working = st.session_state.working_df
    target_classes = sorted(
        str(v) for v in working[selected_target].dropna().unique()
    )
    is_binary = len(target_classes) == 2
    positive_class: str | None = None

    strategy = st.selectbox(
        "Class imbalance",
        [
            "None",
            'class_weight="balanced"',
            "Random oversample",
            "Random undersample",
        ],
    )

    if is_binary:
        stored_pc = st.session_state.get("positive_class")
        default_idx = (
            target_classes.index(stored_pc)
            if stored_pc in target_classes
            else 1
        )
        positive_class = st.selectbox(
            "Positive class",
            target_classes,
            index=default_idx,
            help=(
                "The class treated as 'positive' for Precision, Recall, "
                "F1, ROC-AUC, and PR-AUC. Defaults to the second sorted class."
            ),
        )
        st.session_state.positive_class = positive_class

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
        width="stretch",
    ):
        _train(
            selected_target,
            findings,
            class_weight,
            sampling,
            positive_class=positive_class,
        )

    result_available = st.session_state.latest_training_result is not None
    if st.button(
        "Training result",
        key="training_result_button",
        disabled=not result_available,
        width="stretch",
    ):
        render_training_results()
    if st.button(
        "Feature relevance",
        disabled=not result_available,
        width="stretch",
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
    positive_class: str | None = None,
) -> None:
    try:
        with st.spinner("Training…"):
            result = train_model(
                st.session_state.original_df,
                selected_target,
                class_weight=class_weight,
                sampling=sampling,
                actions=st.session_state.action_log,
                positive_class=positive_class,
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
    st.markdown("**History**")

    undo_stack = st.session_state.get("undo_stack", [])
    redo_stack = st.session_state.get("redo_stack", [])
    undo_col, redo_col = st.columns(2)
    with undo_col:
        if st.button(
            f"↩ Undo ({len(undo_stack)})",
            disabled=len(undo_stack) == 0,
            width="stretch",
            help="Restore the dataset to the state before the last action.",
        ):
            undo_step()
            st.rerun()
    with redo_col:
        if st.button(
            f"↪ Redo ({len(redo_stack)})",
            disabled=len(redo_stack) == 0,
            width="stretch",
            help="Reapply the most recently undone action.",
        ):
            redo_step()
            st.rerun()

    st.divider()
    st.markdown("**Working data**")
    st.download_button(
        "Download working CSV",
        csv_bytes(st.session_state.working_df),
        file_name="datadoctor_working.csv",
        mime="text/csv",
        width="stretch",
    )
    report = build_health_report(
        st.session_state.working_df,
        selected_target,
        findings,
        st.session_state.action_log,
        st.session_state.run_history,
    )
    contract = infer_contract(st.session_state.working_df, selected_target)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Health report",
            health_report_json(report),
            file_name="datadoctor_health_report.json",
            mime="application/json",
            width="stretch",
            help="Download complete audit record and findings in JSON format.",
        )
    with dl_col2:
        st.download_button(
            "Contract",
            contract_json(contract),
            file_name="datadoctor_contract.json",
            mime="application/json",
            width="stretch",
            help="Download inferred schema contract to validate future CSVs.",
        )
    _render_dataset_comparison(contract)
    if st.button("Reset to original", width="stretch"):
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
        st.caption(
            "Contract checks cover required columns, data types, allowed "
            "categories, and missingness above the saved ceiling. A small "
            "missingness change inside that ceiling is expected. "
            "Distribution drift is reported separately."
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
            sections = classify_comparison(contract_issues, drift)
            if (
                not contract_issues
                and not sections["distribution"]
                and not sections["unexpected_missingness"]
            ):
                st.success(
                    "No contract violations or distribution drift "
                    "were found."
                )
            if contract_issues:
                st.markdown("**Contract violations**")
                st.dataframe(
                    pd.DataFrame(contract_issues),
                    hide_index=True,
                    width="stretch",
                )
            if sections["unexpected_missingness"]:
                st.markdown("**Missingness outside the expected range**")
                st.dataframe(
                    pd.DataFrame(sections["unexpected_missingness"]),
                    hide_index=True,
                    width="stretch",
                )
            if sections["distribution"]:
                st.markdown("**Distribution drift**")
                st.dataframe(
                    pd.DataFrame(sections["distribution"]),
                    hide_index=True,
                    width="stretch",
                )
        except (ValueError, OSError) as error:
            st.error(str(error))


def render_locked_target_panel(target_options: list[str]) -> None:
    with st.container(border=True, key="dashboard_locked_target_panel"):
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
    with st.container(border=True, key="dashboard_recovery_panel"):
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
            width="stretch",
            key="recovery_download",
        )
        if st.button(
            "Reset to original",
            width="stretch",
            key="recovery_reset",
        ):
            reset_for_target(selected_target)
            st.rerun()
