from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import DATA_DIR, MODEL_FEATURES, MODELS_DIR, TARGET_COLUMN
from src.data_loader import load_diabetes_dataset, split_features_target
from src.evaluate import evaluate_models, save_evaluation_artifacts
from src.feature_engineering import rank_feature_importance, save_correlation_heatmap, select_features
from src.modeling import train_and_tune_models
from src.preprocessing import balance_training_data, handle_missing_values, scale_features, train_test_data_split


def get_feature_importance_dataframe(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        values = [0.0] * len(feature_names)

    importance_df = pd.DataFrame({"feature": feature_names, "importance": values})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def save_training_artifacts(
    output_dir: Path,
    model,
    scaler,
    imputer,
    best_model_name: str,
    best_params: dict,
    selected_features,
    used_smote: bool,
    feature_ranking_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "best_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(imputer, output_dir / "imputer.joblib")

    feature_ranking_df.to_csv(output_dir / "feature_ranking.csv", index=False)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    metadata = {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "selected_features": selected_features,
        "used_smote": used_smote,
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def run_training_pipeline(csv_path: Path | None = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_diabetes_dataset(csv_path=csv_path)

    dataset_export = DATA_DIR / "diabetes_dataset.csv"
    if not dataset_export.exists():
        df.to_csv(dataset_export, index=False)

    x_data, y_data = split_features_target(df, target_col=TARGET_COLUMN)

    x_selected = select_features(x_data, MODEL_FEATURES)
    selected_df_for_corr = x_selected.copy()
    selected_df_for_corr[TARGET_COLUMN] = y_data
    save_correlation_heatmap(selected_df_for_corr, MODELS_DIR / "correlation_heatmap.png")

    x_train, x_test, y_train, y_test = train_test_data_split(x_selected, y_data)

    x_train_imputed, x_test_imputed, imputer = handle_missing_values(x_train, x_test)
    feature_ranking_df = rank_feature_importance(x_train_imputed, y_train)

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train_imputed, x_test_imputed)

    x_train_balanced, y_train_balanced, used_smote = balance_training_data(x_train_scaled, y_train)

    models, best_params = train_and_tune_models(x_train_balanced, y_train_balanced)

    metrics_df, details, best_model_name = evaluate_models(models, x_test_scaled, y_test)
    save_evaluation_artifacts(MODELS_DIR, metrics_df, details, best_model_name)

    best_model = models[best_model_name]
    feature_importance_df = get_feature_importance_dataframe(best_model, MODEL_FEATURES)

    save_training_artifacts(
        output_dir=MODELS_DIR,
        model=best_model,
        scaler=scaler,
        imputer=imputer,
        best_model_name=best_model_name,
        best_params=best_params,
        selected_features=MODEL_FEATURES,
        used_smote=used_smote,
        feature_ranking_df=feature_ranking_df,
        feature_importance_df=feature_importance_df,
    )

    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print("Artifacts saved in models/ directory.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train diabetes prediction models.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="Optional path to a diabetes CSV dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_file = Path(args.csv_path) if args.csv_path else None
    run_training_pipeline(csv_path=csv_file)
