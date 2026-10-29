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
            padding-top: 1.4rem !important;
            padding-bottom: 3rem !important;
            max-width: 1500px;
        }
        .dd-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.35rem 0 0.8rem;
        }
        .dd-brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        .dd-brand-mark {
            display: grid;
            place-items: center;
            width: 2.8rem;
            height: 2.8rem;
            border-radius: 0.85rem;
            background: linear-gradient(145deg, #1b9568, #0f6847);
            box-shadow: 0 8px 20px rgba(15, 104, 71, 0.22);
            color: white;
            font-size: 1.35rem;
        }
        .dd-hero h1 {
            margin: 0;
            font-size: clamp(1.75rem, 3vw, 2.35rem);
            letter-spacing: -0.04em;
            line-height: 1.05;
        }
        .dd-hero p {
            margin: 0.3rem 0 0;
            color: var(--text-color);
            opacity: 0.68;
            font-size: 0.94rem;
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
            }
            .dd-pill {
                margin-left: 3.6rem;
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
                <div class="dd-brand-mark">✚</div>
                <div>
                    <h1>DataDoctor</h1>
                    <p>Diagnose, improve, and compare your classification data.</p>
                </div>
            </div>
            <div class="dd-pill">Your data stays in your control</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
