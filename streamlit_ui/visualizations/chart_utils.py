"""Shared Altair configuration and utilities for DataDoctor charts."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd


# Colour palette consistent with the DataDoctor green/amber/red theme
PALETTE_CATEGORICAL = [
    "#16845b",
    "#3478c7",
    "#c7873e",
    "#8b5cf6",
    "#e5484d",
    "#06b6d4",
    "#f59e0b",
    "#6366f1",
    "#10b981",
    "#f43f5e",
]

MISSING_COLOUR = "#9ca3af"
COMPACT_HEIGHT = 120
DETAIL_HEIGHT = 220


def base_chart_config() -> dict[str, Any]:
    """Return Altair chart-level config for a clean DataDoctor look."""
    return {
        "view": {"stroke": "transparent"},
        "axis": {
            "labelFontSize": 11,
            "titleFontSize": 11,
            "grid": True,
            "gridOpacity": 0.15,
            "domainOpacity": 0.3,
            "tickOpacity": 0.3,
        },
        "legend": {"labelFontSize": 11, "titleFontSize": 11},
    }


def apply_config(chart: alt.Chart) -> alt.Chart:
    return chart.configure(**base_chart_config())


def missing_fraction(series: pd.Series) -> float:
    return float(series.isna().mean())


def non_missing(series: pd.Series) -> pd.Series:
    return series.dropna()


def freedman_diaconis_bins(series: pd.Series, min_bins: int = 5, max_bins: int = 50) -> int:
    """Compute a reasonable number of histogram bins using the F-D rule."""
    values = non_missing(series)
    n = len(values)
    if n < 4:
        return min_bins
    q75, q25 = values.quantile(0.75), values.quantile(0.25)
    iqr = float(q75 - q25)
    if iqr == 0:
        return min_bins
    bin_width = 2.0 * iqr * (n ** (-1 / 3))
    data_range = float(values.max() - values.min())
    if bin_width <= 0 or data_range <= 0:
        return min_bins
    n_bins = int(round(data_range / bin_width))
    return max(min_bins, min(n_bins, max_bins))
