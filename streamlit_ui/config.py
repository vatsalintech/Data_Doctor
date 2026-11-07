"""Shared configuration for the Streamlit user interface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEMO_FILES = {
    "Messy Telco churn": ROOT / "data" / "Messy" / "telco_messy.csv",
    "Messy Adult income": ROOT / "data" / "Messy" / "adult_messy.csv",
}

DEMO_TARGETS = {
    "Messy Telco churn": "Churn",
    "Messy Adult income": "income",
}

SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
# Cloud-safe limits for free container hosting (~1 GB RAM)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 50_000
MAX_COLUMNS = 150
