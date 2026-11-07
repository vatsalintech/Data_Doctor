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
from data_doctor.detectors import Finding

MIN_BIN_SUPPORT = 5  # Warn when a bin has fewer rows than this


def assess_feature_target_relationship(
    feature: pd.Series,
    target: pd.Series,
    findings: list[Finding],
    feature_name: str,
    target_name: str,
    relevance_pct: float | None = None,
) -> dict[str, str]:
    """Evaluate whether a feature is necessary/important for predicting the target.

    Uses statistical association (class rate variance across categories or
    median shifts across classes) and detector findings to produce a
    plain-English explanation of feature necessity.
    """
    col_findings = [f for f in findings if f.scope == feature_name and f.found]
    leakage = next(
        (f for f in col_findings if f.issue_type in {"target_copy", "suspicious_predictor"}),
        None,
    )
    if leakage:
        return {
            "status": "Unsafe (Target Leakage)",
            "is_necessary": "No — Must Drop",
            "badge_color": "#e5484d",
            "reason": (
                f"This feature almost directly reveals '{target_name}'. Including it "
                f"causes target leakage: the model will achieve an artificially inflated "
                f"score by memorizing the label rather than learning real patterns. It must be dropped."
            ),
        }

    id_like = next((f for f in col_findings if f.issue_type == "id_like"), None)
    if id_like:
        return {
            "status": "Unsafe (Identifier)",
            "is_necessary": "No — Should Drop",
            "badge_color": "#e5484d",
            "reason": (
                f"Almost every row has a unique identifier value. It provides zero "
                f"generalizable predictive signal for '{target_name}' and adds noise. It should be dropped."
            ),
        }

    pair = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(pair) < 10:
        return {
            "status": "Insufficient Data",
            "is_necessary": "Unknown",
            "badge_color": "#9ca3af",
            "reason": f"Not enough non-missing rows to assess the relationship between '{feature_name}' and '{target_name}'.",
        }

    relevance_note = (
        f" In your latest trained model, it accounts for {relevance_pct:.1%} of overall feature influence."
        if (relevance_pct is not None and relevance_pct > 0)
        else ""
    )

    is_numeric = pd.api.types.is_numeric_dtype(feature) and feature.nunique(dropna=True) > 6

    if not is_numeric:
        ct = pd.crosstab(pair["feature"], pair["target"], normalize="index")
        classes = sorted(pair["target"].unique())
        pos_class = classes[-1] if len(classes) == 2 else classes[0]
        rates = ct[pos_class]
        max_cat, max_rate = str(rates.idxmax()), float(rates.max())
        min_cat, min_rate = str(rates.idxmin()), float(rates.min())
        spread = max_rate - min_rate

        if spread >= 0.22:
            return {
                "status": "High Importance",
                "is_necessary": "Yes — Critical / High",
                "badge_color": "#16845b",
                "reason": (
                    f"Target '{target_name}' rate ({pos_class}) varies strongly across categories — "
                    f"from {min_rate:.1%} in '{min_cat}' up to {max_rate:.1%} in '{max_cat}' "
                    f"(a {spread:.1%} spread). This provides strong discriminative power to separate classes.{relevance_note}"
                ),
            }
        elif spread >= 0.10:
            return {
                "status": "Moderate Importance",
                "is_necessary": "Useful",
                "badge_color": "#c7873e",
                "reason": (
                    f"Target '{target_name}' shows noticeable variation across categories "
                    f"(spread: {spread:.1%} between '{min_cat}' and '{max_cat}'). "
                    f"It provides helpful predictive signal when combined with other features.{relevance_note}"
                ),
            }
        else:
            return {
                "status": "Low / Marginal",
                "is_necessary": "Marginal",
                "badge_color": "#3478c7",
                "reason": (
                    f"Target '{target_name}' rate ({pos_class}) is relatively uniform across categories "
                    f"(spread is only {spread:.1%}). By itself, this feature has low standalone influence.{relevance_note}"
                ),
            }
    else:
        classes = sorted(pair["target"].unique())
        medians = {
            cls: float(pair[pair["target"] == cls]["feature"].median())
            for cls in classes
        }
        q75 = float(pair["feature"].quantile(0.75))
        q25 = float(pair["feature"].quantile(0.25))
        iqr = q75 - q25
        data_range = float(pair["feature"].max() - pair["feature"].min())
        denom = iqr if iqr > 0 else (data_range if data_range > 0 else 1.0)
        cls_sorted = sorted(medians.items(), key=lambda x: x[1])
        lo_cls, lo_med = cls_sorted[0]
        hi_cls, hi_med = cls_sorted[-1]
        rel_diff = abs(hi_med - lo_med) / denom

        if rel_diff >= 0.35:
            return {
                "status": "High Importance",
                "is_necessary": "Yes — Critical / High",
                "badge_color": "#16845b",
                "reason": (
                    f"Feature values shift significantly between target classes (median for '{lo_cls}' is "
                    f"{lo_med:,.2f} vs {hi_med:,.2f} for '{hi_cls}'). This distribution difference provides "
                    f"a clear decision boundary for predicting '{target_name}'.{relevance_note}"
                ),
            }
        elif rel_diff >= 0.15:
            return {
                "status": "Moderate Importance",
                "is_necessary": "Useful",
                "badge_color": "#c7873e",
                "reason": (
                    f"Feature values show moderate separation between target classes (median for '{lo_cls}' is "
                    f"{lo_med:,.2f} vs {hi_med:,.2f} for '{hi_cls}'). It adds helpful signal alongside other features.{relevance_note}"
                ),
            }
        else:
            return {
                "status": "Low / Marginal",
                "is_necessary": "Marginal",
                "badge_color": "#3478c7",
                "reason": (
                    f"Feature distributions are very similar across target classes (medians differ by only "
                    f"{abs(hi_med - lo_med):,.2f}). It provides minimal standalone discriminative power.{relevance_note}"
                ),
            }


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

    s_feat = paired["feature"]
    q25 = float(s_feat.quantile(0.25))
    q75 = float(s_feat.quantile(0.75))
    iqr = q75 - q25
    lo_fence = q25 - 1.5 * iqr
    hi_fence = q75 + 1.5 * iqr
    inliers = s_feat[(s_feat >= lo_fence) & (s_feat <= hi_fence)]
    all_span = float(s_feat.max() - s_feat.min())
    inlier_span = float(inliers.max() - inliers.min()) if len(inliers) > 0 else 0.0

    if inlier_span > 0 and (all_span / inlier_span) >= 2.0 and len(inliers) >= 8:
        n_bins = freedman_diaconis_bins(inliers, min_bins=4, max_bins=12)
        _, edges = np.histogram(inliers.to_numpy(dtype=float), bins=n_bins)

        def _get_bin_label(val: float) -> str:
            if val > hi_fence:
                return f"> {edges[-1]:.3g} (outliers)"
            if val < lo_fence:
                return f"< {edges[0]:.3g} (outliers)"
            for i in range(len(edges) - 1):
                if (i == 0 and val >= edges[i] and val <= edges[i + 1]) or (
                    val > edges[i] and val <= edges[i + 1]
                ):
                    return f"{edges[i]:.3g}–{edges[i + 1]:.3g}"
            return f"{edges[-1]:.3g}"

        ordered_labels = []
        if (s_feat < lo_fence).any():
            ordered_labels.append(f"< {edges[0]:.3g} (outliers)")
        for i in range(len(edges) - 1):
            ordered_labels.append(f"{edges[i]:.3g}–{edges[i + 1]:.3g}")
        if (s_feat > hi_fence).any():
            ordered_labels.append(f"> {edges[-1]:.3g} (outliers)")

        paired = paired.copy()
        paired["bin_label"] = paired["feature"].apply(_get_bin_label)
        sort_order = ordered_labels
    else:
        n_bins = freedman_diaconis_bins(paired["feature"], min_bins=4, max_bins=20)
        _, edges = np.histogram(paired["feature"].to_numpy(dtype=float), bins=n_bins)
        paired = paired.copy()
        paired["bin_label"] = pd.cut(
            paired["feature"],
            bins=edges,
            include_lowest=True,
        ).astype(str)
        sort_order = None

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
                sort=sort_order,
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
