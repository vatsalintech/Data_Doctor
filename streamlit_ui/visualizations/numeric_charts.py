"""Numeric column histograms for DataDoctor column cards."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from streamlit_ui.visualizations.chart_utils import (
    COMPACT_HEIGHT,
    DETAIL_HEIGHT,
    MISSING_COLOUR,
    PALETTE_CATEGORICAL,
    freedman_diaconis_bins,
    non_missing,
)


def _bin_data(series: pd.Series) -> pd.DataFrame:
    """Return a tidy DataFrame of histogram bins ready for Altair."""
    values = non_missing(series)
    if len(values) == 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "count", "label"])

    # Separate out infinite values
    finite_mask = np.isfinite(values.to_numpy(dtype=float))
    finite_values = values[finite_mask]
    n_inf = int((~finite_mask).sum())

    n_bins = freedman_diaconis_bins(finite_values)
    counts, edges = np.histogram(finite_values.to_numpy(dtype=float), bins=n_bins)
    rows = [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "count": int(counts[i]),
            "label": f"{edges[i]:.3g}–{edges[i+1]:.3g}",
        }
        for i in range(len(counts))
    ]
    if n_inf > 0:
        rows.append(
            {
                "bin_start": float("nan"),
                "bin_end": float("nan"),
                "count": n_inf,
                "label": "±Inf",
            }
        )
    return pd.DataFrame(rows)


def histogram_chart(
    series: pd.Series,
    *,
    height: int = COMPACT_HEIGHT,
    show_missing: bool = True,
    median_line: bool = True,
    iqr_bounds: bool = False,
    title: str = "",
) -> alt.Chart:
    """Return an Altair histogram for a numeric series."""
    bin_df = _bin_data(series)
    missing_count = int(series.isna().sum())
    total = len(series)

    if bin_df.empty:
        return (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_text(text="No data", align="center")
            .properties(height=height, title=title or "")
        )

    # Main bars — midpoint for positioning
    plot_df = bin_df.copy()
    plot_df["mid"] = (plot_df["bin_start"] + plot_df["bin_end"]) / 2
    plot_df["width"] = plot_df["bin_end"] - plot_df["bin_start"]
    # Exclude ±Inf rows for the main bar chart; they'll be a separate note
    regular = plot_df[plot_df["label"] != "±Inf"].copy()

    base = alt.Chart(regular).mark_bar(
        color=PALETTE_CATEGORICAL[0],
        opacity=0.85,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=alt.X(
            "bin_start:Q",
            bin=alt.BinParams(binned=True),
            title="",
        ),
        x2="bin_end:Q",
        y=alt.Y("count:Q", title="Count"),
        tooltip=[
            alt.Tooltip("label:N", title="Range"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    )

    layers: list[alt.Chart] = [base]

    # Median rule
    if median_line:
        med = float(non_missing(series).median())
        med_df = pd.DataFrame({"median": [med]})
        median_rule = (
            alt.Chart(med_df)
            .mark_rule(color="#f59e0b", strokeWidth=1.5, strokeDash=[4, 2])
            .encode(x=alt.X("median:Q", title=""))
        )
        layers.append(median_rule)

    # IQR bounds
    if iqr_bounds:
        vals = non_missing(series)
        q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
        iqr_v = q3 - q1
        lo, hi = q1 - 1.5 * iqr_v, q3 + 1.5 * iqr_v
        bounds_df = pd.DataFrame({"bound": [lo, hi]})
        bound_rule = (
            alt.Chart(bounds_df)
            .mark_rule(color="#e5484d", strokeWidth=1, strokeDash=[3, 3], opacity=0.7)
            .encode(x=alt.X("bound:Q", title=""))
        )
        layers.append(bound_rule)

    chart = alt.layer(*layers).properties(height=height, title=title or "")

    # Add a missing-values annotation as a subtitle
    if show_missing and missing_count > 0:
        chart = chart.properties(
            title=alt.TitleParams(
                title or "",
                subtitle=f"{missing_count:,} missing ({missing_count / total:.1%})",
                subtitleColor=MISSING_COLOUR,
                subtitleFontSize=10,
            )
        )
    return chart


def compact_histogram(series: pd.Series) -> alt.Chart:
    return histogram_chart(series, height=COMPACT_HEIGHT, median_line=False, iqr_bounds=False)


def detail_histogram(series: pd.Series, column_name: str = "") -> alt.Chart:
    return histogram_chart(
        series,
        height=DETAIL_HEIGHT,
        median_line=True,
        iqr_bounds=True,
        title=f"{column_name} distribution" if column_name else "Distribution",
    )
