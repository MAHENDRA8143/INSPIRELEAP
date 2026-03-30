from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes

from .config import TARGET_COLUMN


def load_diabetes_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        if "diabetic" in df.columns:
            df[TARGET_COLUMN] = (df["diabetic"].str.lower() == "yes").astype(int)
            df = df.drop(columns=["diabetic"])
        elif TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
        return df

    dataset = load_diabetes(as_frame=True)
    df = dataset.frame.copy()

    rename_map = {
        "s1": "cholesterol",
        "s6": "glucose",
    }
    df = df.rename(columns=rename_map)

    threshold = df["target"].median()
    df[TARGET_COLUMN] = (df["target"] >= threshold).astype(int)
    df = df.drop(columns=["target"])

    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    x_data = df.drop(columns=[target_col])
    y_data = df[target_col]
    return x_data, y_data
