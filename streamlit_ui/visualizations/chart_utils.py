"""Shared Altair configuration and utilities for DataDoctor charts."""

from __future__ import annotations

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
