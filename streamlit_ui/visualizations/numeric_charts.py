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


def _bin_data(series: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a tidy DataFrame of histogram bins ready for Altair, plus outlier metadata."""
    values = non_missing(series)
    if len(values) == 0:
        return (
            pd.DataFrame(
                columns=["bin_start", "bin_end", "count", "label", "kind"]
            ),
            {"has_outliers": False},
        )

    # Separate out infinite values
    finite_mask = np.isfinite(values.to_numpy(dtype=float))
    finite_values = values[finite_mask]
    n_inf = int((~finite_mask).sum())

    if len(finite_values) == 0:
        return (
            pd.DataFrame(
                columns=["bin_start", "bin_end", "count", "label", "kind"]
            ),
            {"has_outliers": False},
        )

    q25 = float(finite_values.quantile(0.25))
    q75 = float(finite_values.quantile(0.75))
    iqr = q75 - q25

    meta: dict[str, Any] = {
        "has_outliers": False,
        "n_hi_outliers": 0,
        "n_lo_outliers": 0,
        "max_outlier": None,
        "min_outlier": None,
        "inlier_min": float(finite_values.min()),
        "inlier_max": float(finite_values.max()),
        "q25": q25,
        "q75": q75,
        "median": float(finite_values.median()),
    }

    if iqr > 0 and len(finite_values) >= 8:
        lo_fence = q25 - 1.5 * iqr
        hi_fence = q75 + 1.5 * iqr
        inliers = finite_values[
            (finite_values >= lo_fence) & (finite_values <= hi_fence)
        ]
        outliers_hi = finite_values[finite_values > hi_fence]
        outliers_lo = finite_values[finite_values < lo_fence]

        inlier_span = (
            float(inliers.max() - inliers.min()) if len(inliers) > 0 else 0.0
        )
        all_span = float(finite_values.max() - finite_values.min())

        # If extreme outliers stretch the scale by 2.0x or more, focus chart on inliers
        if inlier_span > 0 and (all_span / inlier_span) >= 2.0:
            meta["has_outliers"] = True
            meta["n_hi_outliers"] = len(outliers_hi)
            meta["n_lo_outliers"] = len(outliers_lo)
            meta["max_outlier"] = (
                float(outliers_hi.max()) if len(outliers_hi) > 0 else None
            )
            meta["min_outlier"] = (
                float(outliers_lo.min()) if len(outliers_lo) > 0 else None
            )
            meta["inlier_min"] = float(inliers.min())
            meta["inlier_max"] = float(inliers.max())
            plot_values = inliers
        else:
            plot_values = finite_values
    else:
        plot_values = finite_values

    n_bins = freedman_diaconis_bins(plot_values)
    counts, edges = np.histogram(plot_values.to_numpy(dtype=float), bins=n_bins)
    bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

    rows: list[dict[str, Any]] = []

    # If there are extreme low outliers, add an underflow indicator bin
    if meta["n_lo_outliers"] > 0:
        underflow_start = float(edges[0] - bin_width)
        underflow_end = float(edges[0])
        rows.append(
            {
                "bin_start": underflow_start,
                "bin_end": underflow_end,
                "count": meta["n_lo_outliers"],
                "label": f"< {edges[0]:.3g} ({meta['n_lo_outliers']} outliers down to {meta['min_outlier']:.3g})",
                "kind": "Outliers",
            }
        )

    # Main inlier bins
    for i in range(len(counts)):
        rows.append(
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(counts[i]),
                "label": f"{edges[i]:.3g}–{edges[i+1]:.3g}",
                "kind": "Data",
            }
        )

    # If there are extreme high outliers, add an overflow indicator bin
    if meta["n_hi_outliers"] > 0:
        overflow_start = float(edges[-1])
        overflow_end = float(edges[-1] + bin_width)
        rows.append(
            {
                "bin_start": overflow_start,
                "bin_end": overflow_end,
                "count": meta["n_hi_outliers"],
                "label": f"> {edges[-1]:.3g} ({meta['n_hi_outliers']} outliers up to {meta['max_outlier']:.3g})",
                "kind": "Outliers",
            }
        )

    if n_inf > 0:
        rows.append(
            {
                "bin_start": float("nan"),
                "bin_end": float("nan"),
                "count": n_inf,
                "label": "±Inf",
                "kind": "Outliers",
            }
        )
    return pd.DataFrame(rows), meta


def histogram_chart(
    series: pd.Series,
    *,
    height: int = COMPACT_HEIGHT,
    show_missing: bool = True,
    median_line: bool = True,
    iqr_bounds: bool = False,
    title: str = "",
    column_name: str = "",
) -> alt.Chart:
    """Return an Altair histogram for a numeric series."""
    bin_df, meta = _bin_data(series)
    missing_count = int(series.isna().sum())
    total = len(series)

    if bin_df.empty:
        return (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_text(text="No data", align="center")
            .properties(height=height, title=title or "")
        )

    # Exclude ±Inf rows for the main bar chart
    regular = bin_df[bin_df["label"] != "±Inf"].copy()

    has_outlier_bars = meta.get("has_outliers", False) and (
        regular["kind"] == "Outliers"
    ).any()

    color_encode = (
        alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["Data", "Outliers"],
                range=[PALETTE_CATEGORICAL[0], "#e5484d"],
            ),
            legend=alt.Legend(title="", orient="top-right"),
        )
        if has_outlier_bars
        else alt.value(PALETTE_CATEGORICAL[0])
    )

    base = alt.Chart(regular).mark_bar(
        opacity=0.85,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=alt.X(
            "bin_start:Q",
            bin=alt.BinParams(binned=True),
            title=column_name or "Value",
            axis=alt.Axis(
                labelFontSize=10,
                titleFontSize=11,
                labelAngle=-25,
            ),
        ),
        x2="bin_end:Q",
        y=alt.Y(
            "count:Q",
            title="Row Count",
            axis=alt.Axis(labelFontSize=10, titleFontSize=11),
        ),
        color=color_encode,
        tooltip=[
            alt.Tooltip("label:N", title=f"{column_name or 'Bin'} Range"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    )

    layers: list[alt.Chart] = [base]

    # Median rule
    if median_line:
        med = meta.get("median", float(non_missing(series).median()))
        med_df = pd.DataFrame(
            {"median": [med], "label": [f"Median: {med:,.2f}"]}
        )
        median_rule = (
            alt.Chart(med_df)
            .mark_rule(color="#f59e0b", strokeWidth=2, strokeDash=[4, 2])
            .encode(
                x=alt.X("median:Q", title=""),
                tooltip=[alt.Tooltip("label:N", title="Statistic")],
            )
        )
        layers.append(median_rule)

    # IQR bounds (Q1 and Q3)
    if iqr_bounds:
        q1 = meta.get("q25", float(non_missing(series).quantile(0.25)))
        q3 = meta.get("q75", float(non_missing(series).quantile(0.75)))
        bounds_df = pd.DataFrame(
            {
                "bound": [q1, q3],
                "label": [f"Q1 (25%): {q1:,.2f}", f"Q3 (75%): {q3:,.2f}"],
            }
        )
        bound_rule = (
            alt.Chart(bounds_df)
            .mark_rule(
                color="#38bdf8", strokeWidth=1.5, strokeDash=[3, 3], opacity=0.85
            )
            .encode(
                x=alt.X("bound:Q", title=""),
                tooltip=[alt.Tooltip("label:N", title="Quartile")],
            )
        )
        layers.append(bound_rule)

    chart = alt.layer(*layers).properties(height=height, title=title or "")

    # Build descriptive subtitle informing user of main range and extreme outliers
    subtitles = []
    if meta.get("has_outliers"):
        total_outliers = meta["n_hi_outliers"] + meta["n_lo_outliers"]
        outlier_val = (
            meta["max_outlier"]
            if meta["max_outlier"] is not None
            else meta["min_outlier"]
        )
        subtitles.append(
            f"Main range ({meta['inlier_min']:,.2f}–{meta['inlier_max']:,.2f}) • {total_outliers:,} extreme outlier(s) up to {outlier_val:,.2f}"
        )
    if show_missing and missing_count > 0:
        subtitles.append(
            f"{missing_count:,} missing ({missing_count / total:.1%})"
        )

    if subtitles:
        chart = chart.properties(
            title=alt.TitleParams(
                title or "",
                subtitle=" • ".join(subtitles),
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
        column_name=column_name,
        title=f"{column_name} distribution" if column_name else "Distribution",
    )
