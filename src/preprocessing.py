"""Data preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, TEST_SIZE


def train_test_data_split(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification."""
    return train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )


def handle_missing_values(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Impute missing numerical values using median."""
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns, index=x_test.index)
    return x_train_imputed, x_test_imputed, imputer


def scale_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numerical features using StandardScaler."""
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    return x_train_scaled, x_test_scaled, scaler


def balance_training_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """Apply SMOTE when the dataset is imbalanced.

    Imbalance criterion: minority/majority ratio below 0.5.
    """
    class_counts = y_train.value_counts()
    majority = class_counts.max()
    minority = class_counts.min()
    ratio = minority / majority

    if ratio >= 0.5:
        return x_train, y_train, False

    smote = SMOTE(random_state=random_state)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    x_resampled_df = pd.DataFrame(x_resampled, columns=x_train.columns)
    y_resampled_series = pd.Series(y_resampled, name=y_train.name)
    return x_resampled_df, y_resampled_series, True
