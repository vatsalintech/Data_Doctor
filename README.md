# DataDoctor

DataDoctor is a beginner-friendly classification CSV health tool. It shows
rule-based data issues, lets the user choose changes, and compares repeated
Logistic Regression runs. It never auto-cleans the uploaded table.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the demo data

Original files under `data/Original/` are never modified.

```bash
python data/build_messy_datasets.py
```

The script writes three deterministic files and an issue manifest under
`data/Messy/`:

- `telco_messy.csv` — target `Churn`
- `adult_messy.csv` — target `income`
- `creditcard_messy.csv` — target `Class`

## Run the detector prototype

```bash
python notebooks/prototype.py
python notebooks/prototype.py data/Messy/adult_messy.csv --target income
python notebooks/prototype.py data/Messy/creditcard_messy.csv --target Class
```

## Run model training directly

```bash
python notebooks/model_training.py
```

The callable used by the app is:

```python
from notebooks.model_training import train_model

result = train_model(df, target="your_target_column")
```

Missing-value imputation, category encoding, and numeric scaling are learned
from training rows only. They do not change the downloadable working CSV.

## Run the app

```bash
streamlit run app.py
```

The app supports upload or demo selection, target selection, diagnosis,
explicit treatments, working-copy download/reset, optional training-only
balancing, saved runs, and between-run comparisons.

## Test

```bash
pytest
```

