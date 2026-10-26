import pandas as pd

df = pd.read_csv("data/demo_messy.csv")

# 1. Basic information
print("DATASET OVERVIEW")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Data types
print("\nCOLUMN DATA TYPES")
print(df.dtypes)

# 3. Missing values, including whitespace-only cells
print("\nMISSING VALUES")
text_df = df.select_dtypes(include=["object", "string"])
blank_cells = text_df.apply(
    lambda col: col.astype("string").str.strip().eq("")
).sum()

missing = df.isna().sum().add(blank_cells, fill_value=0)
print(missing[missing > 0].sort_values(ascending=False))

# 4. Duplicate rows
print("\nDUPLICATES")
print("Exact duplicate rows:", df.duplicated().sum())

# 5. Target distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts(dropna=False))
print("\nChurn proportions:")
print(df["Churn"].value_counts(normalize=True, dropna=False).round(3))

# 6. Numerical columns for later outlier analysis
print("\nNUMERIC COLUMNS")
print(df.select_dtypes(include="number").columns.tolist())