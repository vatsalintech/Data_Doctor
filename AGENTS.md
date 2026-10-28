# DataDoctor — agent brief

Living spec for **DataDoctor**, a classification dataset health tool. Follow this document. Do not re-litigate locked decisions unless the user changes them. The spec will evolve; update this file when a decision changes.

The user is a beginner in AI/ML who knows the basics. Explain choices in plain language. Do not widen scope on your own.

## What this is

An app that takes a classification CSV and lets the user see data problems, choose what to change, download the changed data at any time, and train a model whenever they want so they can see what each change did to the metrics.

It is a **working copy of a dataset**, not a one-shot cleaning report.

The unique part is the training history: baseline vs later runs, with a factual explanation of what changed between runs.

## Locked v1 decisions

- Classification only. No regression, text, images, or time series.
- Start with a few known datasets. The goal is any reasonable classification CSV.
- Never auto-clean. Show every applicable action. Apply only what the user selects.
- Leakage and label checks are **warning signs**, not a general solver. Suspicious labels are flagged for review. Do not silently change labels.
- One model: **Logistic Regression**. No model picker, no AutoML, no deep learning.
- Feature relevance is available after training. Aggregate absolute Logistic Regression coefficients back to source columns, describe them as model influence rather than causation, and let the user explicitly drop selected columns.
- Do not call an LLM to decide issues or write suggestions. Rules decide. Templates explain.
- Streamlit is the UI, built after the prototype script works.
- Stack: pandas, NumPy, scikit-learn, Streamlit.

## Product flow

1. Upload a CSV and preview a few rows.
2. Show basic dataset vitals immediately.
3. User picks the target column from the column names, excluding columns that look like IDs. The full dashboard stays locked until then.
4. Show dataset-level issues and one card per column, with only the actions that apply.
5. User applies actions. The working data, vitals, and action log update immediately.
6. User can download the working data at any time.
7. User trains when they choose. Training receives the working table and the target they already picked. It does not ask for the target again. Do not retrain after every click.
8. Each train is saved as a run. The app compares the latest run with the previous one.

If the user changes the target, reset diagnosis, the action log, and training history. The same file with a different target is a different problem.

## Two layers of text

These look similar in the UI. They come from different places.

### Suggestions ("you could do this")

Come from detectors. Each detector returns a structured finding, not an essay:

- issue type
- column or dataset scope
- severity: high, medium, or low
- evidence (counts, percents, column names)
- suggested action
- one-sentence reason from a template

The user still chooses. Severity is display only. It must not auto-select a fix.

If a check finds nothing, show **No issue found**. Do not hide the column.

### What changed since the last train

Does not come from the detectors. It comes from comparing two saved runs plus the action log.

Always show the facts:

- actions applied between the runs
- columns added or removed
- train row count change
- accuracy and macro F1 before and after
- ROC-AUC too, when the target is binary

Then, if a between-run rule matches, add one interpretation sentence. If no rule matches, show the facts only. Do not guess.

The metric table is the evidence. The sentence is a caption.

## Data the app must remember

- **original**: uploaded CSV, never mutated
- **working copy**: original plus the user's actions; this is what download exports
- **action log**: what the user did, in order
- **run history**: each train's metrics, row/column counts, columns present, open issues, and the action-log position at that time

Provide **Reset to original**. Do not build full per-click undo in v1.

## Cleaning actions vs training-only actions

**Cleaning changes the working CSV** and therefore download and future trains:

- drop duplicate rows
- impute missing values
- cap or remove outliers
- drop a column
- drop rows with bad or conflicting labels

**Training-only actions do not change the download:**

- scaling
- category encoding
- `class_weight="balanced"`

Imbalance resampling (oversample / undersample) is a training option in v1, not a download mutation, unless the user explicitly asks to export a resampled table.

Do not let the user download a scaled or encoded table. Those transforms belong inside the train step.

## Baseline model

"Raw data" often cannot train. Logistic Regression cannot handle missing values or raw categories.

The baseline is the **smallest pipeline that makes the model runnable**:

- basic imputation
- encode categorical columns
- scale numeric columns
- Logistic Regression
- fixed random seed
- stratified train/test split

Call this **Trainable raw**. Later runs are **Trainable raw + the extra cleaning the user has applied**.

## Honest comparison rules

1. Split first. Treat second.
2. Fit anything learned from data (median, mode, scaler, encoder) on train only, then apply to test.
3. Row operations (drop duplicates, drop outliers, resample) happen on the **train** split only.
4. The test set stays as close as possible to the original test split.
5. Dropping a column, including a leaky or ID column, happens on both train and test. Rows stay the same.
6. Same model, same seed, same metrics every run.

If these rules are skipped, the before/after story is not trustworthy.

## Metrics

Always show accuracy, macro F1, and a confusion matrix. Add ROC-AUC for binary targets.

When classes are imbalanced, say that accuracy can look fine while macro F1 is poor.

Also show train rows before vs after, columns dropped, and the between-run caption.

## v1 checks and actions

Six checks only.

### 1. Missing values

Evidence: count and percent per column.

Actions when missing values exist: drop rows with missing values, drop columns that are mostly missing, impute numeric with mean or median, impute categorical with mode, fill with a constant.

### 2. Duplicate records

Evidence: exact duplicate rows, and rows with the same features but different labels.

Actions: drop exact duplicates, drop conflicting-label duplicates, or keep them. Conflicting labels are a review warning. Do not silently change the label.

### 3. Outliers

Method: IQR on numeric columns. Z-score is later.

Actions: do nothing, cap extremes, remove outlier rows.

### 4. Class imbalance

Evidence: class counts and imbalance ratio.

Actions: do nothing, `class_weight="balanced"`, random oversample the training set, random undersample the training set. No SMOTE in v1.

### 5. Suspicious labels

Flag for review: very rare classes, identical features with different labels, label values that look like missing data (`unknown`, `???`, `-1`, blank).

Actions: keep and review, drop those rows, or recode a label only when the user explicitly chooses the new value.

### 6. Leakage warning signs

Flag: target included again as a feature, a column identical or almost identical to the target, ID-like columns (almost every value unique), a feature that predicts the target too perfectly, duplicate rows that could land in both train and test.

Actions: drop the suspicious column, drop the ID-like column, or keep it with a warning.

## Where actions appear

**Dataset issues**, above the column cards:

- duplicate rows
- conflicting labels
- class imbalance

**Column card** for a normal numeric column: type, missing count, min / max / median, outlier count. Actions: impute, cap, drop outlier rows, drop column.

**Categorical column**: type, missing count, unique count, rare categories. Actions: impute with mode, group rare categories into `Other`, drop column.

**ID-like column**: do not lead with mean and standard deviation. Say it looks like an ID and should not be a feature. Actions: drop (recommended) or keep with a warning.

**Leakage suspect**: high severity, drop recommended, or keep with a warning that the score may be fake.

**Target column**: class counts, rare classes, weird labels. The user cannot drop the target. Actions are row/label actions only.

## Severity

- **High**: target leakage signs, a column that duplicates the target, conflicting labels, duplicates that can leak across train and test
- **Medium**: lots of missing values, severe imbalance, many duplicate rows, strong outliers in a numeric feature
- **Low**: a few missing cells, mild outliers, a non-leaky ID, a rare but usable class

Display severity consistently in both dataset and column cards: high is red, medium is yellow, low is blue, and no issue is green.

## Suggestion rulebook

Start with these. Add a rule only after a real case is handled badly.

- Almost all unique values → ID-like → suggest drop column.
- Almost identical to the target → leakage → suggest drop column.
- Numeric missingness under about 30% → suggest median impute.
- Column mostly empty → suggest drop column.
- Numeric IQR outliers → suggest capping before deleting rows.
- Same features, different label → suggest review or drop those rows.
- One class much rarer than the others → suggest `class_weight="balanced"` at train time.
- Target has only one class, or looks like an ID → block training and explain why.

## Between-run rulebook

- Dropped an ID or leaky column and accuracy fell a lot → the previous score was probably too optimistic.
- Dropped an ID or leaky column and the score barely moved → that column was not helping.
- Balanced class weights and macro F1 rose while accuracy stayed flat → better on the rare class.
- Deleted many rows and the score jumped → they may have removed hard examples.
- Imputed missing values and the score barely changed → missingness was not the main problem.
- No rule matches → facts only.

## Files the app should accept

Accept when: it is a CSV, it has enough rows to split, the user picked a target, the target has at least 2 classes, and there is at least one feature column.

Warn or refuse when: the target has one class, the target has too many classes (an ID used as the label), there are almost no rows, every column is empty, or there is no feature left.

## UI

Use a dashboard, not a step-by-step page.

- Top: upload/demo choice, vitals (rows, columns, missing, duplicates), then a collapsed **Preview data** expander showing 10 rows only when opened.
- Main area: dataset-level issues in the top row, then a separated full-width four-column card grid below the controls. Order cards high, medium, low, then no issue. Every card keeps this order together: information, problems, severity, applicable actions. Applying an action refreshes/removes the affected card immediately.
- Put a **Relevance score** toggle to the right of the Column analysis title. Disable it until the first training run. When enabled, show each feature's most recently trained model-influence percentage in parentheses beside its name in the column-card heading; the target has no feature score. Keep those scores unchanged and available after cleaning actions, including column drops, and replace them only after the model is trained again. Reset them when the source or target changes.
- Right control panel: start it directly below the dataset selector, beside the vitals. It contains target selection, Logistic Regression training controls, a green **Training result** button, feature relevance, working-copy download/reset, and the action log. Keep **Training result** directly below Train and above Feature relevance; disable it until the first run.
- Training result opens a dialog that stacks latest run results, confusion matrix, run history, and between-run facts/caption vertically in that order; do not place those titled sections beside each other. Keep Accuracy, Macro F1, and ROC-AUC beside each other within latest run results. Center the confusion-matrix table at a compact width.
- Feature relevance opens a dialog with a non-zoomable 0–1 influence chart. Show every column label on the chart rather than allowing automatic label suppression. Show clickable green Current-column buttons and a separate Drop-columns area; one click moves a column between the lists without a dropdown. Save applies those drops to the working copy and main cards; Cancel applies nothing.

Reference dashboards that show per-column mean and standard deviation are inspiration for the vitals and cards only. Do not copy them. An ID column must not be presented as a normal numeric feature.

Use Streamlit session state so the working copy, action log, and run history survive button clicks.

## Code layout

```text
Data_Doctor/
  AGENTS.md
  app.py                      # Streamlit only, after the prototype script works
  data_doctor/
    detectors.py              # findings, not prose
    treatments.py             # one action at a time
    report.py                 # templates for suggestions and between-run text
  notebooks/
    prototype.py               # demo runner only: load a CSV, pass the target name
    model_training.py          # reusable train_model(df, target)
  data/
    Original/                  # source CSVs; never edit
    Messy/                     # generated demo CSVs plus issue manifest
    build_messy_datasets.py    # deterministic generator; seed 42
  requirements.txt
  README.md
```

`app.py` calls the package. It does not contain the detection or training logic.

## Build order

Do not start with Streamlit.

1. Use three classification demos: mixed Telco churn, mixed Adult income, and numeric credit-card fraud.
2. Build deterministic messy copies with `python data/build_messy_datasets.py`; never edit `data/Original/`.
3. Use `notebooks/prototype.py` as a detector runner that accepts a CSV path and target name.
4. Implement detectors in this order: missing values, duplicates, class imbalance, outliers, suspicious labels, leakage warning signs. Test each on the clean file and the messy file before moving on.
5. Build the minimum trainable pipeline and print accuracy, macro F1, and a confusion matrix.
6. Apply one action (drop the ID column), save the changed CSV, train again, and compare.
7. Only then wrap the same functions in Streamlit.

## Out of scope for v1

- regression and non-tabular data
- SMOTE, Isolation Forest, multiple models, AutoML, deep learning
- an LLM inside the app
- per-click undo beyond reset
- auto-train after every action
- cell-level editing
- user accounts or a database
- downloading scaled or encoded data

## Later

Not part of v1. Do not build these until the six v1 checks and the prototype script work.

- **Stored as text.** Detect a column whose values are numbers but are stored as text, or dates stored as text, and offer a conversion the user can choose. Same idea for other types that show up the same way. Do not convert automatically.
- The clean telco file is the example for the number case: `TotalCharges` is text there because 11 cells are blank, so a normal missing-value count stays at 0. Leave `telco_messy.csv` numeric so the v1 missing-value and outlier checks can see that column.

## Current status

Three immutable originals live in `data/Original/`: Telco churn (`Churn`), Adult income (`income`), and credit-card fraud (`Class`). Run `python data/build_messy_datasets.py` to create deterministic messy copies and `data/Messy/issue_manifest.json`. The manifest is the source of truth for original versus planted issues.

Checks are target-agnostic structured functions in `data_doctor/detectors.py`. `notebooks/prototype.py` is only a command-line demo runner. Training is intentionally separate in `notebooks/model_training.py` as `train_model(df, target)`; the UI imports it directly. No detector or training function may hardcode demo target or feature names.

`app.py` is the one-page Streamlit v1 app. It keeps original data, a working copy, action log, and run history in session state. The user chooses the target before target-dependent diagnosis and trains only on demand.

Type detection for numbers or dates stored as text remains on the Later list.

When a product decision changes, update this file in the same session so the next agent does not follow a stale rule.
