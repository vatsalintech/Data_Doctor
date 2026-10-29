"""Shared configuration for the Streamlit user interface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEMO_FILES = {
    "Messy Telco churn": ROOT / "data" / "Messy" / "telco_messy.csv",
    "Messy Adult income": ROOT / "data" / "Messy" / "adult_messy.csv",
    "Messy credit-card fraud": (
        ROOT / "data" / "Messy" / "creditcard_messy.csv"
    ),
}

DEMO_TARGETS = {
    "Messy Telco churn": "Churn",
    "Messy Adult income": "income",
    "Messy credit-card fraud": "Class",
}

SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_ROWS = 500_000
MAX_COLUMNS = 250
