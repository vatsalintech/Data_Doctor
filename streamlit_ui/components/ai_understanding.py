"""AI Data Understanding UI components.

Renders the privacy-first AI configuration panel, dataset-level AI findings card,
and column-level AI understanding subsection inside each column card.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_doctor.ai_understanding import (
    AIClientError,
    AIUnderstandingEngine,
    AIUnderstandingResult,
    ColumnUnderstanding,
    detect_sensitive_column,
)


def get_ai_engine() -> AIUnderstandingEngine:
    """Retrieve or initialize the AI engine singleton in session state."""
    if "ai_engine" not in st.session_state:
        st.session_state.ai_engine = AIUnderstandingEngine()
    return st.session_state.ai_engine


# ─────────────────────────────────────────────────────────────────────────────
# 1. AI Understanding Entry Point & Configuration Panel
# ─────────────────────────────────────────────────────────────────────────────


def render_ai_understanding_section(
    df: pd.DataFrame,
    target: str | None = None,
) -> None:
    """Render the AI Data Understanding configuration, privacy, and trigger panel."""
    with st.expander("AI Data Understanding (Optional)", expanded=False):
        st.markdown(
            "Get an independent AI interpretation of your dataset and its columns. "
            "AI findings provide a second perspective alongside rule-based diagnosis "
            "and **will never modify your data, auto-clean, or alter existing severity levels**."
        )

        engine = get_ai_engine()
        client = engine.get_client()
        if hasattr(client, "model") and getattr(client, "model", None) and not getattr(client, "should_fail", None) and client.__class__.__name__ == "GeminiAIClient":
            st.caption(f"🟢 **Live AI Connected:** Google Gemini (`{client.model}`)")
        else:
            st.caption(
                "🟡 **Offline Simulator Mode:** `GEMINI_API_KEY` not detected. "
                "Add your key to `.streamlit/secrets.toml` or set `export GEMINI_API_KEY=...` to connect live AI."
            )

        st.markdown("---")
        st.markdown("#### Privacy & Data Sharing Disclosure")
        st.info(
            "**What will be shared if you run AI understanding:**\n"
            "- Overall vitals: row count, column count, column names, inferred data types.\n"
            "- Python summary statistics: min, max, median, mean, common categories, and missingness.\n"
            "- Compact representative samples: approximately 15–20 non-missing values per included column.\n\n"
            "⚠️ *Note: Representative sampling extracts real sample values and does not guarantee anonymization. "
            "Direct identifiers or sensitive columns can be excluded below.*"
        )

        # Detect candidate sensitive columns for default exclusion recommendation
        sensitive_candidates: list[str] = []
        for col in df.columns:
            is_sens, _ = detect_sensitive_column(df[col], str(col))
            if is_sens:
                sensitive_candidates.append(str(col))

        col_left, col_right = st.columns([1, 1], gap="medium")
        with col_left:
            st.markdown("**Column Exclusions**")
            current_excluded = st.session_state.get(
                "ai_excluded_columns", sensitive_candidates
            )
            # Ensure excluded columns exist in current dataframe
            valid_excluded = [c for c in current_excluded if c in df.columns]

            selected_exclusions = st.pills(
                "Exclude columns from AI analysis",
                options=df.columns.tolist(),
                default=valid_excluded,
                selection_mode="multi",
                help=(
                    "Click to toggle columns on or off. Excluded columns will not have sample values "
                    "sent to the AI service. Columns matching identifier or sensitive heuristics are "
                    "pre-selected by default."
                ),
            )
            st.session_state.ai_excluded_columns = selected_exclusions or []

            if sensitive_candidates:
                st.caption(
                    f"Heuristics detected potential identifiers or PII: {', '.join(sensitive_candidates)}"
                )

        with col_right:
            st.markdown("**Dataset Description (Optional)**")
            desc_val = st.text_area(
                "Provide business context or domain",
                value=st.session_state.get("ai_user_description", ""),
                placeholder="e.g. Telecommunications customer billing and subscription records...",
                help="Helps the AI model understand domain terminology, units, and intended goals.",
                height=90,
            )
            st.session_state.ai_user_description = desc_val

        st.markdown("---")
        consent_col, button_col = st.columns([3, 1], gap="medium")
        with consent_col:
            consent = st.checkbox(
                "I confirm that I have reviewed the shared data summary and consent to sending "
                "the specified context and non-excluded sample values to the external AI provider.",
                value=st.session_state.get("ai_consent", False),
                key="ai_consent_checkbox",
            )
            st.session_state.ai_consent = consent

        with button_col:
            has_existing = st.session_state.get("ai_understanding_result") is not None
            btn_label = (
                "Rerun AI Understanding" if has_existing else "Run AI Understanding"
            )
            run_btn = st.button(
                btn_label,
                type="primary",
                disabled=not consent,
                width="stretch",
            )

        if run_btn:
            _execute_ai_understanding(df, target)


def _execute_ai_understanding(
    df: pd.DataFrame,
    target: str | None = None,
) -> None:
    """Execute AI understanding engine with progress reporting."""
    engine = get_ai_engine()
    excluded = set(st.session_state.get("ai_excluded_columns", []))
    description = st.session_state.get("ai_user_description") or None

    progress_bar = st.progress(0, text="Preparing dataset context and representative samples...")
    status_text = st.empty()

    def update_progress(done: int, total: int) -> None:
        pct = min(1.0, done / max(1, total))
        progress_bar.progress(
            pct,
            text=f"Interpreting columns with AI ({done}/{total} columns completed)...",
        )

    try:
        status_text.info("Generating dataset-level and batched column semantic analysis...")
        result = engine.run_understanding(
            df,
            target=target,
            user_description=description,
            excluded_columns=excluded,
            progress_callback=update_progress,
            force_refresh=True,
        )
        st.session_state.ai_understanding_result = result
        st.session_state.ai_context_fingerprint = result.metadata.get("fingerprint")
        st.session_state.ai_is_stale = False
        progress_bar.empty()
        status_text.success("AI Data Understanding completed successfully!")
        st.rerun()
    except AIClientError as client_err:
        progress_bar.empty()
        status_text.error(f"AI Understanding failed: {client_err}")
        st.caption(
            "All existing rule-based diagnosis, cleaning, and model training features "
            "remain fully accessible."
        )
    except Exception as ex:
        progress_bar.empty()
        status_text.error(f"Unexpected error during AI analysis: {ex}")
        st.caption("Existing rule-based features are unaffected.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset-Level AI Findings Card (Displayed Above Column Cards)
# ─────────────────────────────────────────────────────────────────────────────


def render_ai_dataset_understanding_card(
    result: AIUnderstandingResult | None,
) -> None:
    """Display the overall dataset interpretation above the column analysis grid."""
    if result is None:
        return

    du = result.dataset_understanding
    is_stale = st.session_state.get("ai_is_stale", False)

    with st.expander("🤖 AI Dataset Understanding", expanded=True):
        if is_stale:
            st.warning(
                "⚠️ **Dataset or target modified**: Actions were applied or the target changed "
                "since this AI interpretation was run. These findings describe a prior dataset state. "
                "You can rerun analysis anytime from the AI Data Understanding panel above."
            )

        # Overview Header
        head_col, conf_col = st.columns([4, 1])
        with head_col:
            st.markdown(f"**Likely Meaning:** {du.description}")
            st.markdown(f"**Possible Purpose:** {du.possible_purpose}")
        with conf_col:
            conf_color = (
                "#22c55e"
                if du.confidence == "high"
                else "#3b82f6"
                if du.confidence == "medium"
                else "#f59e0b"
            )
            st.markdown(
                f"<div style='text-align:right;'>"
                f"<span style='background-color:{conf_color}22; color:{conf_color}; "
                f"padding:4px 10px; border-radius:12px; font-weight:600; font-size:0.85rem;'>"
                f"Confidence: {du.confidence.title()}</span></div>",
                unsafe_allow_html=True,
            )

        # Dataset Composition
        if du.composition:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown("**Major Information Categories:**")
            tags_html = " ".join(
                [
                    f"<span style='background-color:rgba(128,128,128,0.15); padding:3px 8px; "
                    f"border-radius:6px; font-size:0.82rem; margin-right:6px;'>{c}</span>"
                    for c in du.composition
                ]
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        # Target Assessment (if present)
        if du.target_assessment:
            ta = du.target_assessment
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"**Target Column Assessment (`{ta.target_name}`)**")
                st.markdown(f"- **Represents:** {ta.target_meaning}")
                st.markdown(f"- **Purpose alignment:** {ta.consistency_with_purpose}")
                for obs in ta.observations:
                    st.markdown(f"- *Observation:* {obs}")

        # Cross-column Observations
        if du.observations:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown("**Dataset-Level Observations & Cross-Column Relationships:**")
            for obs in du.observations:
                with st.container(border=True):
                    st.markdown(f"**{obs.title}** (Confidence: *{obs.confidence}*)")
                    st.markdown(obs.description)
                    if obs.limitation:
                        st.caption(f"Contextual limitation: {obs.limitation}")

        # Limitations
        if du.limitations:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown("**Contextual Limitations & Missing Information:**")
            for lim in du.limitations:
                st.markdown(f"- {lim}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Column-Level AI Findings Subsection (Inside Each Column Card)
# ─────────────────────────────────────────────────────────────────────────────


def render_column_ai_understanding(
    column_name: str,
    result: AIUnderstandingResult | None,
) -> None:
    """Render collapsible AI Understanding subsection inside an individual column card."""
    with st.expander("🤖 AI Understanding", expanded=False):
        if result is None:
            st.caption(
                "AI interpretation has not been run yet. Expand the 'AI Data Understanding' "
                "panel above to run independent analysis."
            )
            return

        is_stale = st.session_state.get("ai_is_stale", False)
        if is_stale:
            st.caption("⚠️ *Dataset was modified after this AI analysis was run.*")

        excluded_cols = result.metadata.get("excluded_columns", [])
        if column_name in excluded_cols:
            st.info("🔒 This column was excluded from AI sample analysis per user privacy settings.")

        col_result: ColumnUnderstanding | None = result.columns.get(column_name)
        if col_result is None:
            st.caption("No AI interpretation is available for this column.")
            return

        # Meaning & Semantic Type
        st.markdown(f"**Likely Meaning:** {col_result.likely_meaning}")

        sem_label = col_result.semantic_type.replace("_", " ").title()
        sec_tags = (
            f" ({', '.join(col_result.secondary_semantic_types)})"
            if col_result.secondary_semantic_types
            else ""
        )
        st.markdown(f"**Semantic Type:** `{sem_label}`{sec_tags}")

        # Expected Data Structure
        es = col_result.expected_structure
        with st.container(border=True):
            st.markdown("**Expected Structure:**")
            st.markdown(f"- Logical Type: *{es.logical_type}*")
            if es.expected_format:
                st.markdown(f"- Format: *{es.expected_format}*")
            if es.expected_units:
                st.markdown(f"- Units: *{es.expected_units}*")
            if es.possible_range:
                st.markdown(f"- Range: *{es.possible_range}*")
            if es.expected_constraints:
                st.markdown(f"- Constraints: {', '.join(es.expected_constraints)}")

        # Observed Consistency
        oc = col_result.observed_consistency
        status_colors = {
            "consistent": "#22c55e",
            "potentially_inconsistent": "#eab308",
            "insufficient_information": "#3b82f6",
            "ambiguous": "#a855f7",
        }
        color = status_colors.get(oc.status, "#6b7280")
        badge_html = (
            f"<div style='margin-bottom:6px;'>"
            f"<span style='background-color:{color}22; color:{color}; "
            f"padding:3px 8px; border-radius:10px; font-weight:600; font-size:0.8rem;'>"
            f"Consistency: {oc.status.replace('_', ' ').title()}</span></div>"
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        for obs_item in oc.observations:
            st.markdown(f"- {obs_item}")

        # Confidence & Limitations
        st.caption(f"Confidence: **{col_result.confidence.title()}**")
        if col_result.limitations:
            st.markdown("**Limitations:**")
            for lim_item in col_result.limitations:
                st.caption(f"- {lim_item}")
