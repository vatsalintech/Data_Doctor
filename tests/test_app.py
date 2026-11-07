from pathlib import Path

from streamlit.testing.v1 import AppTest

from data_doctor.detectors import Finding
from streamlit_ui.components.dataset_issues import dataset_issue_findings

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_starts_with_collapsed_preview_and_severity_order():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    assert not app.exception
    top_expanders = [
        item.label
        for item in app.expander
        if item.label in {"Preview data", "Compare a new CSV"}
    ]
    assert top_expanders == [
        "Preview data",
        "Compare a new CSV",
    ]

    headings = [item.value for item in app.markdown]
    assert headings.index("#### ChurnCopy") < headings.index(
        "#### Contract"
    )
    assert headings.index("#### Contract") < headings.index(
        "#### customerID"
    )
    assert headings.index("#### customerID") < headings.index(
        "#### gender"
    )
    assert any(
        "inside that ceiling is expected" in caption.value
        for caption in app.caption
    )


def test_feature_column_moves_to_drop_list_with_one_click():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    next(
        button
        for button in app.button
        if button.label == "Train Logistic Regression"
    ).click().run(timeout=30)
    saved_relevance = dict(
        app.session_state["last_feature_relevance"]
    )
    next(
        toggle
        for toggle in app.toggle
        if toggle.label == "Relevance score"
    ).set_value(True).run(timeout=30)
    assert any(
        heading.value.startswith("#### customerID (")
        and heading.value.endswith("%)")
        for heading in app.markdown
    )
    next(
        button
        for button in app.button
        if button.label == "Feature relevance"
    ).click().run(timeout=30)

    assert not app.multiselect
    assert any(
        "one-standard-deviation" in caption.value
        and "Correlated columns" in caption.value
        for caption in app.caption
    )
    next(
        button
        for button in app.button
        if button.label == "customerID"
    ).click().run(timeout=30)
    assert app.session_state["feature_drop_selection"] == [
        "customerID"
    ]
    next(
        button
        for button in app.button
        if button.label == "Apply action"
    ).click().run(timeout=30)
    assert app.session_state["latest_training_result"] is None
    assert (
        dict(app.session_state["last_feature_relevance"])
        == saved_relevance
    )
    assert any(
        heading.value.startswith("#### customerID (")
        and heading.value.endswith("%)")
        for heading in app.markdown
    )
    result_button = next(
        button
        for button in app.button
        if button.label == "Training result"
    )
    assert result_button.disabled
    assert any(
        "Results are out of date" in caption.value
        for caption in app.caption
    )


def test_training_result_unlocks_and_opens_in_dialog():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    model_buttons = [
        button
        for button in app.button
        if button.label
        in {
            "Train Logistic Regression",
            "Training result",
            "Feature relevance",
        }
    ]
    assert [button.label for button in model_buttons] == [
        "Train Logistic Regression",
        "Training result",
        "Feature relevance",
    ]
    assert model_buttons[1].disabled

    model_buttons[0].click().run(timeout=30)
    result_button = next(
        button
        for button in app.button
        if button.label == "Training result"
    )
    assert not result_button.disabled
    result_button.click().run(timeout=30)

    assert any(
        heading.value == "Latest run results"
        for heading in app.subheader
    )
    assert {metric.label for metric in app.metric} >= {
        "Accuracy",
        "Macro F1",
    }
    assert any(
        "ROC-AUC positive class: Yes" in caption.value
        for caption in app.caption
    )
    assert any(
        "Macro F1 is shown only" in caption.value
        for caption in app.caption
    )
    first_run = app.session_state["run_history"][0]
    assert first_run.get("precision") is not None
    assert first_run.get("recall") is not None
    assert first_run.get("f1") is not None
    assert not app.exception


def test_readiness_is_tentative_and_impact_preview_is_absent():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    assert all(
        button.label not in {"Preview", "Preview impact"}
        for button in app.button
    )
    assert any(
        "Checks passed so far" in (m.help or "")
        for m in app.markdown
    ) or any(
        "Checks passed so far" in message.value
        for message in app.info
    )
    assert "impact_preview" not in app.session_state


def test_upload_mode_explains_that_large_files_can_take_time():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    app.radio[0].set_value("Upload CSV").run()
    assert any(
        "Large files can take a few minutes to train" in caption.value
        for caption in app.caption
    )


def test_stored_text_columns_are_promoted_to_dataset_issues():
    finding = Finding(
        issue_type="stored_as_text",
        scope="amount",
        severity="medium",
        found=True,
        evidence={"inferred_type": "number"},
        suggested_action="convert_to_number",
        reason="Stored as text.",
    )
    assert dataset_issue_findings([finding]) == [finding]


def test_column_cards_have_separate_distribution_and_target_analysis_expanders():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    assert not app.exception

    expander_labels = [item.label for item in app.expander]
    assert "📊 Column chart distribution" in expander_labels
    assert "🎯 Target analysis" in expander_labels
    assert "📊 Charts & Target Analysis" not in expander_labels

    imbalance_select = next(
        s for s in app.selectbox if s.label == "Class imbalance"
    )
    assert imbalance_select.options == [
        "None",
        'class_weight="balanced"',
        "Random oversample",
        "Random undersample",
    ]


def test_column_expand_button_opens_dialog_popup_with_two_columns():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    assert not app.exception

    # Find the expand button for the first column (e.g. ChurnCopy)
    expand_btn = next(
        b for b in app.button if b.key and b.key.startswith("expand_col_")
    )
    col_name = expand_btn.key.replace("expand_col_", "")
    expand_btn.click().run(timeout=30)
    assert not app.exception

    # Subheader with column name inside dialog
    assert any(col_name in s.value for s in app.subheader)
    # Check chart heading and action section inside dialog
    assert any("Column distribution" in m.value for m in app.markdown)
    assert any("Actions & Status" in m.value for m in app.markdown)


def test_numeric_histogram_and_target_distribution_robust_to_extreme_outliers():
    import pandas as pd
    import numpy as np
    from streamlit_ui.visualizations.numeric_charts import _bin_data, detail_histogram
    from streamlit_ui.visualizations.target_analysis import numeric_target_distribution

    # Series with 100 normal values in [10, 100] and 2 extreme outliers at 10,000
    normal = list(range(10, 100))
    series = pd.Series(normal + [10000.0, 10000.0])

    bin_df, meta = _bin_data(series)
    assert meta["has_outliers"] is True
    assert meta["n_hi_outliers"] == 2
    assert meta["inlier_min"] == 10.0
    assert meta["inlier_max"] == 99.0
    # Last bin is an outlier overflow bin
    assert any(row["kind"] == "Outliers" for _, row in bin_df.iterrows())
    # The normal bins span ~10 to ~99, not 10,000
    data_bins = bin_df[bin_df["kind"] == "Data"]
    assert data_bins["bin_end"].max() < 200

    # Ensure chart builds without error
    chart = detail_histogram(series, column_name="Amount")
    spec = chart.to_dict()
    assert spec is not None

    # Test target distribution with extreme outliers
    target = pd.Series([0 if v < 55 else 1 for v in normal] + [1, 1])
    target_chart = numeric_target_distribution(
        series, target, feature_name="Amount", target_name="Target"
    )
    assert target_chart is not None
    target_spec = target_chart.to_dict()
    assert target_spec is not None





