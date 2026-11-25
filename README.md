# DataDoctor 🩺

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datadoctorai.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google Gemini](https://img.shields.io/badge/Gemini%20API-Flash%20Lite-8E75B2?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **An interactive, human-in-the-loop data quality workbench for tabular classification datasets.**  
> 🔗 **Live Demo:** [datadoctorai.streamlit.app](https://datadoctorai.streamlit.app/)

---

## 🎯 What is DataDoctor

DataDoctor is a **decision-support tool for data scientists**, not an automatic data cleaner. It is scoped to **tabular classification** (a discrete target with at least two classes). Continuous regression targets are not supported.

Most data cleaning tools fall into one of two extremes: passive static report generators (like Pandas Profiling) or dangerous black-box "auto-cleaners" that silently impute values, delete rows, and contaminate held-out test splits.

DataDoctor takes a different approach. It **diagnoses your dataset using deterministic rules, surfaces every detected issue with a plain-language explanation, and waits for you to decide what to do.** You choose each action explicitly. Nothing is changed until you apply it.

The core idea is to show you **whether each cleaning decision changed held-out model performance**. After you apply actions, you can re-train Logistic Regression or HistGradientBoostingClassifier and compare Accuracy, Macro F1, ROC-AUC, and more. On the hold-out path, a 400-repetition bootstrap confidence interval tells you whether the change is larger than ordinary test-set noise. A flat or slightly moving score is a valid result — many fills change very little on the demo files, and removing a leaked target copy often lowers the score on purpose.

**What it is not:** DataDoctor does not auto-clean your data. It does not make changes without your approval. It does not build a production ML pipeline, tune hyperparameters, or train regression models.

---

## 📸 Core Features

```
       Raw / Messy CSV
              │
              ▼
   ┌────────────────────────────────────────────────────────┐
   │                  DataDoctor Workbench                  │
   │                                                        │
   │  ┌───────────────────────┐   ┌───────────────────────┐ │
   │  │  Deterministic Rules  │   │  AI Understanding     │ │
   │  │  7 Tabular Detectors  │   │  Gemini Semantic Map  │ │
   │  └───────────┬───────────┘   └───────────┬───────────┘ │
   │              │                           │             │
   │              ▼                           ▼             │
   │  ┌──────────────────────────────────────────────────┐  │
   │  │   Interactive Dashboard & Atomic Interventions   │  │
   │  └───────────────────────┬──────────────────────────┘  │
   │                          │                             │
   │                          ▼                             │
   │  ┌──────────────────────────────────────────────────┐  │
   │  │        Honest ML Pipeline & Model Auditing       │  │
   │  │ • Group-Aware Split    • Train-Only Fits         │  │
   │  │ • Bootstrap Range      • Feature Relevance       │  │
   │  │ • Logistic Regression  • HistGradientBoosting    │  │
   │  │ • Hold-out 80/20       • Stratified K-Fold CV    │  │
   │  └──────────────────────────────────────────────────┘  │
   └────────────────────────────────────────────────────────┘
              │                               │
              ▼                               ▼
       Cleaned Working CSV            Portable JSON Contract
        (Always Exportable)            & PSI Drift Monitor
              │                               │
              ▼                               ▼
      Trained Model Pipeline         Python Cleaning Script
      (.joblib scikit-learn)         (.py standalone code)
```

### 1. Honest ML Evaluation Rules
- **Split First, Treat Second**: Transformers (scalers, encoders, median imputers) and row-level cleaning operations (outlier clipping, row drops, balancing) are learned strictly on the **training split**, leaving the test / validation split unpolluted by fit-time statistics.
- **Group-Aware Splitting**: Identical feature rows are kept on one side of the split so duplicate rows cannot leak across train and test.
- **Hold-out split (80/20)**: Default evaluation. Fixed seed, same rows every Train click for the same file and target.
- **Stratified K-Fold Cross-Validation (5-Fold / 10-Fold)**: Group-aware stratified folds, Mean ± Std across folds, and out-of-fold diagnostics. The downloadable model is then fit on the full cleaned training table.
- **Paired Bootstrap Check (400 Repetitions)**: On the hold-out path only, resamples the same held-out rows 400 times and shows a 95% range for the change in Accuracy and Macro F1. When that range includes zero, the app calls the change normal test-set wobble. K-fold uses fold Mean ± Std instead.

### 2. Two Training Models — Side by Side
Both models produce the same metric keys so you can compare how each responds to the same cleaning decisions.

| | Logistic Regression | HistGradientBoosting |
|:---|:---|:---|
| **Type** | Linear | Gradient-boosted trees |
| **Missing values** | Median / mode imputed before training | Handled natively — no imputation needed |
| **Categoricals** | OneHotEncoded (top 50 categories) | OrdinalEncoded (top 50 categories) |
| **Feature relevance** | Normalised absolute coefficient weights | Permutation feature importance (mean accuracy drop on held-out samples) |
| **Best for** | Understanding linear relationships, fast baseline | Non-linear patterns, native missing handling |

### 3. 7 Deterministic Anomaly Detectors
- **Missing Values**: Quantifies missing rates per column (including common text stand-ins) and suggests median/mode imputation, constant filling, or column pruning.
- **Exact & Conflicting Duplicates**: Identifies identical records and highlights conflicting-label rows for manual review.
- **Outliers (IQR Fences)**: Flags numerical outliers and suggests bounding/capping before dropping rows.
- **Class Imbalance**: Computes class ratios and provides training-only interventions (`class_weight="balanced"`, random oversampling, random undersampling).
- **Suspicious Labels**: Detects rare classes, near-constant labels, and missing-like strings (`unknown`, `???`, blanks) with targeted recoding tools. Spelling variants of the same answer are flagged for manual review, not auto-merged.
- **Target Leakage & ID Columns**: Flags columns identical to the target, high-lift target proxies, and high-cardinality IDs.
- **Features Stored as Text**: Safely parses numeric/datetime values mistakenly encoded as strings and provides explicit 1-click conversions.

### 4. Full Metric Suite
Every model run reports:
- Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall
- ROC-AUC and Average Precision (PR-AUC) for binary problems
- Threshold slider for binary classification — adjust the decision boundary live (does not refit)
- Per-class Precision, Recall, F1, and Support
- Confusion matrix
- Slice diagnostics — per-group accuracy and F1 for categorical columns
- Run history table comparing previous runs

### 5. Dataset Contracts and Drift Checks
- Exports a JSON contract capturing column types, missingness ceilings, and category bounds.
- Checks a later CSV against that contract, including a **Population Stability Index (PSI)** comparison for numeric columns.

### 6. Decoupled AI Understanding Layer
- Integrates Google Gemini (`gemini-flash-lite-latest`) to provide plain-language domain summaries, business context, and column semantic roles.
- **Strictly Separated**: Rules and math diagnose and clean data; the LLM strictly explains domain context and never modifies data.
- **Offline fallback**: With no API key, the app uses a built-in mock and does not call Gemini. Diagnosis and training keep working.

### 7. Exportable Artifacts & Code Generation
- **Trained Model File (`.joblib`)**: Download the fitted scikit-learn Pipeline (preprocessing + model) from the controls or training result dialog.
- **Reproducible Python Script (`.py`)**: Download standalone Python that replays the cleaning steps applied so far (`python datadoctor_cleaning.py input.csv output.csv`).
- **Cleaned Data & Audit Logs**: Export the working CSV, schema contract JSON, and health audit report JSON at any time.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Core Libraries** | pandas, NumPy, scikit-learn, joblib |
| **Frontend / UI** | Streamlit, Altair, HTML/CSS |
| **LLM Semantic Layer** | Google Gemini API via `requests`, Pydantic v2 |
| **Testing** | pytest, Streamlit AppTest framework |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 installed.

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/vatsalintech/Data_Doctor.git
cd Data_Doctor

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Configure Gemini API Key

DataDoctor works out of the box with offline mock summaries. To enable live Gemini AI domain insights:

**Option A — Environment Variable:**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

**Option B — Streamlit Secrets:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` and paste your key:
```toml
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-flash-lite-latest"
```
*(Note: `.streamlit/secrets.toml` is included in `.gitignore` and is never committed to Git).*

### 3. Launch the Application

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

The demo menu provides two built-in demo datasets: **Telco Customer Churn** and **Adult Census Income** (located in `data/Messy/`). Rebuilding them from the immutable originals in `data/Original/` is optional:

```bash
python data/build_messy_datasets.py
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Deploy from [share.streamlit.io](https://share.streamlit.io/) with main file `app.py` and Python 3.12 from `runtime.txt`.
2. Optionally add `GEMINI_API_KEY` and `GEMINI_MODEL` under **Settings → Secrets**.

The 10 MB upload cap, 50,000-row ceiling, and 8-step undo history exist because the free tier has about 1 GB of RAM. They are hosting limits, not limits of the diagnosis or training logic. Datasets under about **20,000 rows** stay responsive on that free tier.

---

## 🧪 Testing & Verification

Tests cover detectors, treatments, hold-out and K-fold training, the bootstrap check, contracts, exports, Streamlit interactions, and the AI fallback. Run them locally:

```bash
pytest
```

---

## 📁 Repository Structure

```text
Data_Doctor/
├── app.py                            # Streamlit entry point
├── runtime.txt                       # Cloud environment Python specification (3.12)
├── requirements.txt                  # Dependency version ranges
├── data/
│   ├── Original/                     # Immutable source datasets (Telco, Adult)
│   ├── Messy/                        # Generated messy demo CSVs & issue manifest
│   └── build_messy_datasets.py       # Deterministic demo generator
├── data_doctor/
│   ├── code_export.py                # Model serialization (.joblib) & Python code generator
│   ├── contracts.py                  # Inferred dataset contracts & PSI drift monitoring
│   ├── detectors.py                  # 7 deterministic tabular issue detectors
│   ├── model_training.py             # Leakage-safe training pipeline (LR & HGB)
│   ├── treatments.py                 # Atomic, reproducible cleaning actions
│   ├── report.py                     # Suggestion templates & between-run comparisons
│   └── ai_understanding/             # Decoupled Gemini semantic understanding layer
│       ├── ai_client.py              # Gemini client with MockAIClient fallback
│       ├── dataset_analyzer.py       # Domain purpose & dataset-level interpretation
│       ├── dataset_context.py        # Shared dataset context passed to AI calls
│       ├── column_analyzer.py        # Column-level semantic classification
│       ├── sample_generator.py       # Representative value sampling for prompts
│       ├── prompts.py                # Prompt templates for Gemini calls
│       ├── response_schema.py        # Pydantic v2 structured schemas
│       ├── result_validator.py       # Validates & normalises AI responses
│       └── cache_manager.py          # Response caching to prevent duplicate API calls
├── scripts/
│   └── diagnose_cli.py               # Command-line inspection tool for tabular datasets
├── streamlit_ui/                     # Modular Streamlit frontend
│   ├── dashboard.py                  # Page layout & orchestration
│   ├── data_sources.py               # File uploader & demo dataset loaders
│   ├── state.py                      # Session state, undo/redo stack
│   ├── actions.py                    # UI action options & inputs
│   ├── config.py                     # App-wide UI configuration constants
│   ├── styles.py                     # Page theme and visual styling
│   ├── visualizations/               # Altair chart modules
│   │   ├── categorical_charts.py     # Bar & frequency charts for categorical columns
│   │   ├── numeric_charts.py         # Histograms & outlier bin charts
│   │   ├── target_analysis.py        # Target distribution & class-balance visuals
│   │   └── chart_utils.py            # Shared Altair helpers & colour scales
│   └── components/                   # Reusable UI components
│       ├── controls.py               # Sidebar controls & global action buttons
│       ├── columns.py                # Column list & summary cards
│       ├── column_detail.py          # Per-column detail panel
│       ├── column_actions.py         # Action inputs for individual columns
│       ├── column_charts.py          # Inline column-level chart rendering
│       ├── dataset_issues.py         # Dataset-level issue cards & interventions
│       ├── ai_understanding.py       # AI insights panel & rendering
│       ├── dialogs.py                # Modal dialogs (training results, exports)
│       └── common.py                 # Shared component helpers & micro-widgets
└── tests/                            # pytest suite
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
