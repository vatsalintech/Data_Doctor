"""Categorical column frequency charts for DataDoctor column cards."""

from __future__ import annotations

import altair as alt
import pandas as pd

from streamlit_ui.visualizations.chart_utils import (
    COMPACT_HEIGHT,
    DETAIL_HEIGHT,
    MISSING_COLOUR,
    PALETTE_CATEGORICAL,
    non_missing,
)

TOP_N = 10


def _category_counts(series: pd.Series, top_n: int = TOP_N) -> pd.DataFrame:
    """Return tidy category-count DataFrame, capping at top_n + 'Other'."""
    total = len(series)
    missing_count = int(series.isna().sum())
    value_counts = series.value_counts(dropna=True)
    top = value_counts.head(top_n)
    other_count = int(value_counts.iloc[top_n:].sum()) if len(value_counts) > top_n else 0

    rows = [
        {
            "category": str(cat),
            "count": int(cnt),
            "pct": cnt / total,
            "is_other": False,
        }
        for cat, cnt in top.items()
    ]
    if other_count > 0:
        rows.append(
            {
                "category": f"Other ({len(value_counts) - top_n} more)",
                "count": other_count,
                "pct": other_count / total,
                "is_other": True,
            }
        )
    if missing_count > 0:
        rows.append(
            {
                "category": "(missing)",
                "count": missing_count,
                "pct": missing_count / total,
                "is_other": False,
            }
        )
    return pd.DataFrame(rows)


def _colour_scale(df: pd.DataFrame) -> alt.Scale:
    colours = []
    for _, row in df.iterrows():
        if row["category"] == "(missing)":
            colours.append(MISSING_COLOUR)
        elif row["is_other"]:
            colours.append("#d1d5db")
        else:
            colours.append(PALETTE_CATEGORICAL[0])
    return alt.Scale(domain=df["category"].tolist(), range=colours)


def frequency_chart(
    series: pd.Series,
    *,
    height: int = DETAIL_HEIGHT,
    title: str = "",
    column_name: str = "",
    top_n: int = TOP_N,
) -> alt.Chart:
    """Return a horizontal bar chart of category frequencies."""
    df = _category_counts(series, top_n=top_n)
    if df.empty:
        return (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_text(text="No data", align="center")
            .properties(height=height, title=title or "")
        )

    colour_scale = _colour_scale(df)
    calculated_height = max(180, len(df) * 26 + 35)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "category:N",
                sort="-x",
                title=column_name or "Category",
                axis=alt.Axis(
                    labelLimit=160,
                    labelOverlap=False,
                    labelFontSize=10,
                    titleFontSize=11,
                    labelPadding=6,
                ),
            ),
            x=alt.X(
                "count:Q",
                title="Row Count",
                axis=alt.Axis(labelFontSize=10, titleFontSize=11),
            ),
            color=alt.Color(
                "category:N",
                scale=colour_scale,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=calculated_height, title=title or "")
    )
    return chart


def compact_frequency_chart(series: pd.Series) -> alt.Chart:
    return frequency_chart(series, height=COMPACT_HEIGHT, top_n=8)


def detail_frequency_chart(series: pd.Series, column_name: str = "") -> alt.Chart:
    return frequency_chart(
        series,
        height=DETAIL_HEIGHT,
        top_n=TOP_N,
        column_name=column_name,
        title=f"{column_name} — top categories" if column_name else "Top categories",
    )
