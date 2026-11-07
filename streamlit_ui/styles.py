"""Page setup, shared styles, and branding."""

import streamlit as st


def configure_page() -> None:
    """Configure the page and apply the DataDoctor visual theme."""
    st.set_page_config(
        page_title="DataDoctor",
        page_icon="🩺",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --dd-green: #16845b;
            --dd-green-dark: #0f6847;
            --dd-green-soft: rgba(22, 132, 91, 0.10);
            --dd-blue: #3478c7;
            --dd-blue-dark: #255f9f;
            --dd-blue-soft: rgba(49, 116, 190, 0.10);
            --dd-red: #c74646;
            --dd-red-dark: #9f3434;
            --dd-amber-soft: rgba(217, 143, 24, 0.12);
            --dd-red-soft: rgba(205, 65, 65, 0.10);
            --dd-border: rgba(128, 128, 128, 0.22);
            --dd-shadow: 0 6px 22px rgba(15, 40, 30, 0.06);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 8% 0%,
                    rgba(22, 132, 91, 0.07),
                    transparent 24rem
                ),
                var(--background-color);
        }
        .block-container {
            padding-top: 3.2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1500px;
        }
        .dd-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.6rem 0 0.8rem;
        }
        .dd-brand {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .dd-title-row {
            display: flex !important;
            align-items: center !important;
            gap: 0.65rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .dd-brand-mark {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2.25rem !important;
            height: 2.25rem !important;
            flex-shrink: 0 !important;
            border-radius: 0.65rem !important;
            background: linear-gradient(145deg, #1b9568, #0f6847) !important;
            box-shadow: 0 4px 14px rgba(15, 104, 71, 0.22) !important;
            color: white !important;
            font-size: 1.2rem !important;
            line-height: 1 !important;
        }
        .dd-hero h1,
        .dd-title,
        [data-testid="stMarkdownContainer"] .dd-title,
        [data-testid="stMarkdownContainer"] .dd-hero h1 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: clamp(1.75rem, 3vw, 2.35rem) !important;
            letter-spacing: -0.04em !important;
            line-height: 1.1 !important;
            font-weight: 750 !important;
        }
        .dd-hero p,
        .dd-subtitle,
        [data-testid="stMarkdownContainer"] .dd-subtitle,
        [data-testid="stMarkdownContainer"] .dd-hero p {
            margin: 0.15rem 0 0 !important;
            padding: 0 !important;
            color: var(--text-color) !important;
            opacity: 0.68 !important;
            font-size: 0.94rem !important;
            line-height: 1.3 !important;
        }
        .dd-pill {
            flex: 0 0 auto;
            padding: 0.4rem 0.75rem;
            border: 1px solid rgba(22, 132, 91, 0.24);
            border-radius: 999px;
            background: var(--dd-green-soft);
            color: var(--dd-green);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--dd-border) !important;
            border-radius: 0.9rem !important;
            box-shadow: var(--dd-shadow);
            background: color-mix(
                in srgb,
                var(--background-color) 97%,
                var(--dd-green) 3%
            );
        }
        /* Controls panel: compact vertical rhythm */
        [class*="st-key-dashboard_controls_panel"] {
            display: flex !important;
            flex-direction: column !important;
        }
        [class*="st-key-dashboard_controls_panel"] hr {
            margin: 0.35rem 0 !important;
        }
        [class*="st-key-dashboard_controls_panel"] div[data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }
        [class*="st-key-dashboard_controls_panel"] [data-testid="stButton"] button,
        [class*="st-key-dashboard_controls_panel"] [data-testid="stDownloadButton"] button {
            min-height: 2.15rem !important;
            padding: 0.2rem 0.5rem !important;
            font-size: 0.84rem !important;
        }
        [class*="st-key-dashboard_controls_panel"] [data-testid="stSelectbox"] {
            margin-bottom: 0 !important;
        }
        [class*="st-key-dashboard_controls_panel"] [data-testid="stExpander"] {
            margin-bottom: 0.1rem !important;
        }
        [class*="st-key-dashboard_controls_panel"] p {
            margin-bottom: 0.1rem !important;
        }
        [class*="st-key-dashboard_"] {
            height: auto !important;
            max-height: none !important;
        }
        [class*="st-key-dashboard_"] > div[data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
        }
        .dd-card-header {
            min-height: 3.5rem;
            margin-bottom: 0.25rem;
        }
        .dd-card-info {
            min-height: 5.6rem;
            font-size: 0.88rem;
            line-height: 1.45;
            padding-bottom: 0.25rem;
        }
        .dd-card-problems {
            min-height: 5.8rem;
            max-height: 8.5rem;
            overflow-y: auto;
            padding-bottom: 0.25rem;
        }
        .dd-card-actions {
            min-height: 5.2rem;
            margin-bottom: 0.4rem;
        }
        [data-testid="stMetric"] {
            min-height: 6.4rem;
            border: 1px solid var(--dd-border);
            border-radius: 0.9rem;
            padding: 0.95rem 1.05rem;
            background: var(--background-color);
            box-shadow: var(--dd-shadow);
        }
        [data-testid="stMetricValue"] {
            color: var(--dd-green);
            font-size: 1.8rem;
            font-weight: 750;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.025em;
            text-transform: uppercase;
            white-space: normal;
        }
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            min-height: 2.55rem;
            border-radius: 0.65rem;
            font-weight: 650;
            transition:
                transform 120ms ease,
                border-color 120ms ease,
                box-shadow 120ms ease;
        }
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            border-color: var(--dd-green);
            box-shadow: 0 4px 12px rgba(15, 104, 71, 0.12);
            transform: translateY(-1px);
        }
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(
                135deg,
                var(--dd-green),
                var(--dd-green-dark)
            ) !important;
            border-color: var(--dd-green-dark) !important;
        }
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 0.65rem !important;
        }
        /* Ensure selectbox dropdown menus do not truncate option labels */
        div[data-baseweb="popover"] {
            min-width: max-content !important;
            max-width: min(92vw, 450px) !important;
        }
        div[data-baseweb="popover"] ul[role="listbox"],
        div[data-baseweb="popover"] [data-baseweb="menu"] {
            min-width: max-content !important;
        }
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] span {
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: unset !important;
            padding-right: 1.25rem !important;
        }
        [data-testid="stExpander"] {
            border-color: var(--dd-border);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--dd-border);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        h2, h3 {
            letter-spacing: -0.025em;
        }
        h3 {
            margin-top: 1.4rem !important;
        }
        /* Ensure Column analysis heading text and Relevance score toggle align on the exact same center line */
        [data-testid="stHorizontalBlock"]:has(#column-analysis) {
            align-items: center !important;
        }
        [data-testid="stHorizontalBlock"]:has(#column-analysis) > [data-testid="stColumn"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="stHorizontalBlock"]:has(#column-analysis) > [data-testid="stColumn"]:first-child > div {
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stHorizontalBlock"]:has(#column-analysis) > [data-testid="stColumn"]:last-child > div {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
        }
        [data-testid="stHorizontalBlock"]:has(#column-analysis) h3,
        [data-testid="stHorizontalBlock"]:has(#column-analysis) h3#column-analysis,
        [data-testid="stHorizontalBlock"]:has(#column-analysis) [data-testid="stHeadingWithActionElements"] {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stHorizontalBlock"]:has(#column-analysis) [data-testid="stCheckbox"],
        [data-testid="stHorizontalBlock"]:has(#column-analysis) [data-testid="stCheckbox"] > label {
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
            display: flex !important;
            align-items: center !important;
        }
        /* Segmented pagination bar styling */
        [class*="st-key-col_page_"] button {
            min-height: 2.25rem !important;
            height: 2.25rem !important;
            border-radius: 0.65rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 0 0.5rem !important;
        }
        /* Column card header expand button */
        [class*="st-key-expand_col_"] button {
            min-height: 1.85rem !important;
            height: 1.85rem !important;
            padding: 0 !important;
            font-size: 1.05rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 0.5rem !important;
        }
        hr {
            border-color: var(--dd-border) !important;
        }
        .control-title {
            font-size: 1.22rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin-bottom: 0.7rem;
        }
        .dd-severity {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            width: fit-content;
            padding: 0.28rem 0.62rem;
            border: 1px solid transparent;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 750;
            line-height: 1;
        }
        .dd-severity::before {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: currentColor;
            content: "";
        }
        .dd-severity-high {
            color: #c73f3f;
            background: var(--dd-red-soft);
            border-color: rgba(205, 65, 65, 0.20);
        }
        .dd-severity-medium {
            color: #b87309;
            background: var(--dd-amber-soft);
            border-color: rgba(217, 143, 24, 0.22);
        }
        .dd-severity-low {
            color: #3174be;
            background: var(--dd-blue-soft);
            border-color: rgba(49, 116, 190, 0.20);
        }
        .dd-severity-none {
            color: var(--dd-green);
            background: var(--dd-green-soft);
            border-color: rgba(22, 132, 91, 0.20);
        }
        h4 {
            min-height: 2.35rem;
            margin: 0 !important;
            overflow-wrap: anywhere;
        }
        .dd-issue-title {
            min-height: 3rem;
            margin: 0 0 0.35rem;
            font-size: 1.08rem;
            font-weight: 750;
            line-height: 1.25;
            letter-spacing: -0.015em;
            overflow-wrap: break-word;
        }
        .dd-section-label {
            margin: 0.2rem 0 0.1rem;
            color: var(--text-color);
            opacity: 0.58;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        /* ── Sticky dialog header & internal scroll ───────────────────────── */
        /* 1. Prevent outer backdrop overlay from scrolling modal card off-screen */
        [data-testid="stDialog"],
        div.stDialog,
        div[class*="ep9mgll0"] {
            overflow: hidden !important;
            align-items: center !important;
            justify-content: center !important;
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
        }
        /* 2. Constrain dialog card to viewport height and layout as flex column */
        [data-testid="stDialog"] > div:first-child,
        div.stDialog > div:first-child,
        div[class*="ep9mgll1"] {
            max-height: calc(100vh - 3rem) !important;
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
            position: relative !important;
        }
        /* 3. The inner role="dialog" wrapper */
        [data-testid="stDialog"] [role="dialog"],
        div.stDialog [role="dialog"],
        div[class*="ep9mgll2"] {
            display: flex !important;
            flex-direction: column !important;
            max-height: calc(100vh - 3rem) !important;
            height: 100% !important;
            overflow: hidden !important;
            position: relative !important;
        }
        /* 4. Sticky Title / Header pinned at top */
        [data-testid="stDialog"] [slot="title"],
        [data-testid="stDialog"] h2,
        div.stDialog [slot="title"],
        div.stDialog h2,
        div[class*="ep9mgll5"],
        [data-testid="stDialog"] [data-testid="stDialogTitle"] {
            flex-shrink: 0 !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 99 !important;
            background: var(--background-color) !important;
            margin: 0 !important;
            padding: 1.1rem 3.5rem 0.85rem 1.5rem !important;
            border-bottom: 1px solid var(--dd-border) !important;
        }
        /* 5. Sticky Close Button (X) pinned at top right */
        [data-testid="stDialog"] button[aria-label="Close"],
        div.stDialog button[aria-label="Close"],
        button[class*="ep9mgll4"] {
            position: absolute !important;
            top: 1rem !important;
            right: 1.25rem !important;
            z-index: 100 !important;
            background: transparent !important;
            cursor: pointer !important;
        }
        /* 6. The dialog content body container - THIS is what scrolls! */
        [data-testid="stDialog"] [role="dialog"] > div:last-child,
        div.stDialog [role="dialog"] > div:last-child,
        div[class*="ep9mgll6"] {
            flex: 1 1 auto !important;
            overflow-y: auto !important;
            min-height: 0 !important;
            padding: 1.25rem 1.5rem 1.5rem !important;
        }
        [data-testid="stDialog"] button[kind="secondary"] {
            background: #123d2a;
            color: #8ff0b8;
            border-color: #287a50;
        }
        [data-testid="stDialog"] button[kind="secondary"]:hover {
            background: #185238;
            color: #b9ffd3;
            border-color: #36a469;
        }
        .st-key-training_result_button button:not(:disabled) {
            background: #1f7a4d;
            color: white;
            border-color: #2d9b65;
        }
        .st-key-training_result_button button:not(:disabled):hover {
            background: #286f4d;
            color: white;
            border-color: #3eaf76;
        }
        .st-key-train_model_button button {
            background: linear-gradient(
                135deg,
                var(--dd-blue),
                var(--dd-blue-dark)
            ) !important;
            color: white !important;
            border-color: var(--dd-blue-dark) !important;
        }
        .st-key-train_model_button button:hover {
            background: var(--dd-blue-dark) !important;
            border-color: var(--dd-blue) !important;
            box-shadow: 0 4px 12px rgba(37, 95, 159, 0.22) !important;
        }
        [class*="st-key-drop_feature_"] button {
            background: linear-gradient(
                135deg,
                var(--dd-red),
                var(--dd-red-dark)
            ) !important;
            color: white !important;
            border-color: var(--dd-red-dark) !important;
        }
        [class*="st-key-drop_feature_"] button:hover {
            background: var(--dd-red-dark) !important;
            border-color: var(--dd-red) !important;
            box-shadow: 0 4px 12px rgba(159, 52, 52, 0.22) !important;
        }
        @media (max-width: 700px) {
            .dd-hero {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.6rem;
            }
            .dd-pill {
                margin-left: 0;
            }
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the dashboard heading."""
    st.markdown(
        """
        <div class="dd-hero">
            <div class="dd-brand">
                <div class="dd-title-row">
                    <span class="dd-brand-mark">✚</span>
                    <h1 class="dd-title">DataDoctor</h1>
                </div>
                <p class="dd-subtitle">Diagnose, improve, and compare your classification data.</p>
            </div>
            <div class="dd-pill">Your data stays in your control</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
