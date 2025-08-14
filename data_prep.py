"""
data_prep.py

Robust loader for the UCI "Breast Cancer Wisconsin (Original)" dataset.
- Accepts common file names or prompts upload in Colab
- Drops ID column
- Handles '?' missing values
- Maps class 2->0 (benign), 4->1 (malignant)
- Standardizes features (StandardScaler)
- Returns X_train, X_test, y_train, y_test
"""
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_FILENAMES = [
    "Breast_Cancer_Data.csv",
    "breast-cancer-wisconsin.data",
    "breast-cancer-wisconsin.data.txt",
    "breast-cancer-wisconsin.data.csv",
    "breast-cancer-wisconsin.data"
]

def _find_file(provided_path=None):
    # If caller provided exact path and it exists, return it
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return str(p)

    # Look for common filenames in current working dir
    for name in DEFAULT_FILENAMES:
        p = Path.cwd() / name
        if p.exists():
            return str(p)

    # Search any file with 'breast' in name
    for p in Path.cwd().glob("*breast*.*"):
        return str(p)

    # Not found
    return None

def _prompt_upload():
    # Only works in Colab; if not available raise FileNotFoundError
    try:
        from google.colab import files
        print("Please upload your dataset file (e.g. 'breast-cancer-wisconsin.data').")
        uploaded = files.upload()
        # return the first uploaded filename
        for fn in uploaded:
            return fn
    except Exception:
        raise FileNotFoundError("Dataset not found in working directory and upload prompt unavailable.")

def load_data(path=None, test_size=0.25, random_state=42, standardize=True):
    # locate the file
    fp = _find_file(path)
    if not fp:
        fp = _prompt_upload()
        if not fp:
            raise FileNotFoundError("Dataset file could not be located or uploaded.")

    # Read the file - these rows are comma-separated lines like:
    # 1000025,5,1,1,1,2,1,3,1,1,2
    # We'll attempt a straightforward read and fall back to robust cleaning if parse fails.
    try:
        df = pd.read_csv(fp, header=None, sep=',', engine='python')
    except Exception:
        # fallback: keep only lines starting with a digit and reparse
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip() and ln.strip()[0].isdigit()]
        if not lines:
            raise ValueError(f"No numeric data lines found in file {fp}. Please ensure you uploaded the correct dataset.")
        cleaned = StringIO("\n".join(lines))
        # try comma sep first
        try:
            df = pd.read_csv(cleaned, header=None, sep=',', engine='python')
        except Exception:
            cleaned.seek(0)
            df = pd.read_csv(cleaned, header=None, delim_whitespace=True, engine='python')

    # Ensure we have at least 11 columns; if more, take first 11
    if df.shape[1] < 11:
        raise ValueError(f"Parsed data has {df.shape[1]} columns; expected at least 11. Check dataset format.")
    if df.shape[1] > 11:
        df = df.iloc[:, :11]

    # Assign UCI column names
    df.columns = [
        "Sample_code_number",
        "Clump_Thickness",
        "Uniformity_Cell_Size",
        "Uniformity_Cell_Shape",
        "Marginal_Adhesion",
        "Single_Epithelial_Cell_Size",
        "Bare_Nuclei",
        "Bland_Chromatin",
        "Normal_Nucleoli",
        "Mitoses",
        "Class"
    ]

    # Drop ID column
    df = df.drop(columns=["Sample_code_number"])

    # Handle missing values ('?') and coerce to numeric
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)

    # Extract X and y; map classes 2 -> 0, 4 -> 1
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({2: 0, 4: 1}).astype(int)

    # Stratified train-test split (keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Return numpy arrays for convenience
    return X_train, X_test, y_train.values, y_test.values, before, after

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, before, after = load_data()
    print(f"Loaded data. Total rows before dropna: {before}, after dropna: {after}")
    print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
