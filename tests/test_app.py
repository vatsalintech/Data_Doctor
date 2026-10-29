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
    assert [item.label for item in app.expander] == [
        "Preview data",
        "Pre-train readiness",
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


def test_readiness_is_tentative_and_impact_preview_is_absent():
    app = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=30
    ).run()
    assert all(
        button.label not in {"Preview", "Preview impact"}
        for button in app.button
    )
    assert any(
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

