"""Model evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_models(models: Dict[str, object], x_test, y_test) -> Tuple[pd.DataFrame, Dict[str, dict], str]:
    """Evaluate models and return a metrics DataFrame, detailed metrics map, and best model name."""
    rows = []
    details: Dict[str, dict] = {}

    for model_name, model in models.items():
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
            }
        )

        details[model_name] = {
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }

    results_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["model"]
    return results_df, details, best_model_name


def save_evaluation_artifacts(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    details: Dict[str, dict],
    best_model_name: str,
) -> None:
    """Persist evaluation metrics and ROC data for app use."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    metrics_json_path = output_dir / "metrics.json"
    metrics_payload = {
        "best_model": best_model_name,
        "models": metrics_df.to_dict(orient="records"),
        "details": details,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    best_roc = details[best_model_name]["roc_curve"]
    roc_data_path = output_dir / "best_model_roc.npz"
    np.savez(
        roc_data_path,
        fpr=np.array(best_roc["fpr"]),
        tpr=np.array(best_roc["tpr"]),
    )
