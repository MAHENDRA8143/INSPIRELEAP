"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes

from .config import TARGET_COLUMN


def load_diabetes_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load a diabetes dataset into a pandas DataFrame.

    If a CSV path is provided and exists, it is loaded directly.
    Otherwise, the sklearn diabetes dataset is loaded and converted
    into a binary classification target.
    """
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        # Handle Yes/No target column conversion
        if "diabetic" in df.columns:
            df[TARGET_COLUMN] = (df["diabetic"].str.lower() == "yes").astype(int)
            df = df.drop(columns=["diabetic"])
        elif TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
        return df

    dataset = load_diabetes(as_frame=True)
    df = dataset.frame.copy()

    # Make feature names friendlier for downstream UI and reports.
    rename_map = {
        "s1": "cholesterol",  # total serum cholesterol
        "s6": "glucose",      # blood sugar level proxy
    }
    df = df.rename(columns=rename_map)

    # Convert regression target to binary classification label.
    threshold = df["target"].median()
    df[TARGET_COLUMN] = (df["target"] >= threshold).astype(int)
    df = df.drop(columns=["target"])

    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target."""
    x_data = df.drop(columns=[target_col])
    y_data = df[target_col]
    return x_data, y_data
