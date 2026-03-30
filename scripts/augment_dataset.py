from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

np.random.seed(42)


def augment_dataset():
    df = pd.read_csv(DATA_DIR / "Diabetes_Final_Data_V2.csv")

    base_cholesterol = 150 + (df["age"] * 0.5) + (df["bmi"] * 2) + (df["glucose"] * 10)
    cholesterol = base_cholesterol + np.random.normal(0, 15, len(df))
    df["cholesterol"] = np.clip(cholesterol, 100, 450).round(1)

    activity_base = 5 - ((df["bmi"] - 20) * 0.15) - ((df["age"] - 40) * 0.02)
    activity_level = activity_base + np.random.normal(0, 0.5, len(df))
    df["physical_activity_level"] = np.clip(activity_level, 0, 7).round(1)

    output_path = DATA_DIR / "Diabetes_Final_Data_Augmented.csv"
    df.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nNew columns added:")
    print(f"  - cholesterol (range: {df['cholesterol'].min():.1f} - {df['cholesterol'].max():.1f})")
    print(
        f"  - physical_activity_level (range: {df['physical_activity_level'].min():.1f} - {df['physical_activity_level'].max():.1f})"
    )
    print(f"\nFirst few rows:")
    print(df[["age", "bmi", "glucose", "cholesterol", "physical_activity_level", "diabetic"]].head(10))


if __name__ == "__main__":
    augment_dataset()
