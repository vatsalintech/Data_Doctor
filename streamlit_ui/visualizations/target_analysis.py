"""Feature-vs-target distribution charts for DataDoctor."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from streamlit_ui.visualizations.chart_utils import (
    DETAIL_HEIGHT,
    PALETTE_CATEGORICAL,
    freedman_diaconis_bins,
    non_missing,
)

MIN_BIN_SUPPORT = 5  # Warn when a bin has fewer rows than this


def _target_classes(target_series: pd.Series) -> list[str]:
    return sorted(str(v) for v in target_series.dropna().unique())


def numeric_target_distribution(
    feature: pd.Series,
    target: pd.Series,
    *,
    feature_name: str = "",
    target_name: str = "",
    positive_class: str | None = None,
) -> alt.Chart | None:
    """Stacked bar: for each numeric bin, show class proportions within it."""
    paired = (
        pd.DataFrame({"feature": feature, "target": target.astype(str)})
        .dropna(subset=["feature", "target"])
    )
    if len(paired) < 10:
        return None

    finite_mask = np.isfinite(paired["feature"].to_numpy(dtype=float))
    paired = paired[finite_mask]
    if len(paired) < 10:
        return None

    n_bins = freedman_diaconis_bins(paired["feature"], min_bins=4, max_bins=20)
    _, edges = np.histogram(paired["feature"].to_numpy(dtype=float), bins=n_bins)
    paired = paired.copy()
    paired["bin_label"] = pd.cut(
        paired["feature"],
        bins=edges,
        include_lowest=True,
    ).astype(str)

    classes = _target_classes(target)
    colour_range = PALETTE_CATEGORICAL[: len(classes)]

    group = (
        paired.groupby(["bin_label", "target"], observed=False)
        .size()
        .reset_index(name="count")
    )
    bin_totals = group.groupby("bin_label", observed=False)["count"].transform("sum")
    group["pct"] = group["count"] / bin_totals.values

    # Warn caption rows for thin bins (< MIN_BIN_SUPPORT)
    thin_bins = (
        group.groupby("bin_label", observed=False)["count"]
        .sum()
        .lt(MIN_BIN_SUPPORT)
    )
    has_thin = thin_bins.any()

    chart = (
        alt.Chart(group)
        .mark_bar()
        .encode(
            x=alt.X(
                "bin_label:N",
                sort=None,
                title=feature_name or "Feature range",
                axis=alt.Axis(labelAngle=-30, labelLimit=100),
            ),
            y=alt.Y("pct:Q", title="Proportion", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "target:N",
                scale=alt.Scale(domain=classes, range=colour_range),
                title=target_name or "Target",
            ),
            tooltip=[
                alt.Tooltip("bin_label:N", title="Range"),
                alt.Tooltip("target:N", title="Class"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(
            height=DETAIL_HEIGHT,
            title=f"{feature_name} × {target_name}" if feature_name and target_name else "Feature vs Target",
        )
    )
    return chart


def categorical_target_distribution(
    feature: pd.Series,
    target: pd.Series,
    *,
    feature_name: str = "",
    target_name: str = "",
    top_n: int = 10,
) -> alt.Chart | None:
    """Stacked bar: for each category (top-N), show class proportions."""
    paired = (
        pd.DataFrame({"feature": feature.astype(str), "target": target.astype(str)})
        .dropna()
    )
    if len(paired) < 10:
        return None

    top_cats = (
        paired["feature"].value_counts().head(top_n).index.tolist()
    )
    paired = paired.copy()
    paired["feature"] = paired["feature"].where(
        paired["feature"].isin(top_cats), "Other"
    )
    classes = _target_classes(target)
    colour_range = PALETTE_CATEGORICAL[: len(classes)]

    group = (
        paired.groupby(["feature", "target"])
        .size()
        .reset_index(name="count")
    )
    bin_totals = group.groupby("feature")["count"].transform("sum")
    group["pct"] = group["count"] / bin_totals.values

    chart = (
        alt.Chart(group)
        .mark_bar()
        .encode(
            x=alt.X(
                "feature:N",
                sort="-y",
                title=feature_name or "Category",
                axis=alt.Axis(labelAngle=-30, labelLimit=120),
            ),
            y=alt.Y("pct:Q", title="Proportion", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "target:N",
                scale=alt.Scale(domain=classes, range=colour_range),
                title=target_name or "Target",
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Category"),
                alt.Tooltip("target:N", title="Class"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(
            height=max(DETAIL_HEIGHT, len(top_cats) * 22 + 40),
            title=f"{feature_name} × {target_name}" if feature_name and target_name else "Feature vs Target",
        )
    )
    return chart
