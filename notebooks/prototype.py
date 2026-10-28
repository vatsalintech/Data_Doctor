"""Command-line demo for DataDoctor's reusable detectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_doctor.detectors import (  # noqa: E402
    dataset_vitals,
    diagnose,
    finding_rows,
    prepare_data,
)

DEFAULT_FILE = PROJECT_ROOT / "data" / "Messy" / "telco_messy.csv"
DEFAULT_TARGET = "Churn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print DataDoctor findings for a classification CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_FILE,
        help="CSV path (defaults to the messy telco demo).",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Name of the classification target column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = (
        args.csv
        if args.csv.is_absolute()
        else (Path.cwd() / args.csv).resolve()
    )
    df = prepare_data(pd.read_csv(path))

    print("DATASET OVERVIEW")
    print(pd.Series(dataset_vitals(df)).to_string())
    print(f"selected_target       {args.target}")

    findings = diagnose(df, args.target)
    print("\nFINDINGS")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(finding_rows(findings).to_string(index=False))


if __name__ == "__main__":
    main()