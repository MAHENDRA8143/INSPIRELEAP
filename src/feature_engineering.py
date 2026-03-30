"""Feature engineering and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create and save a correlation heatmap image."""
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def rank_feature_importance(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Rank features using mutual information."""
    scores = mutual_info_classif(x_train, y_train, random_state=42)
    importance_df = pd.DataFrame({"feature": x_train.columns, "mutual_info": scores})
    importance_df = importance_df.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    return importance_df


def select_features(x_data: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Select model input features by name."""
    return x_data[selected_features].copy()
