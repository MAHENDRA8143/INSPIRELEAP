"""Streamlit web app for diabetes risk prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def load_artifacts():
    """Load trained model and metadata from disk."""
    model_path = MODELS_DIR / "best_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    imputer_path = MODELS_DIR / "imputer.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    metrics_path = MODELS_DIR / "metrics.csv"
    importance_path = MODELS_DIR / "feature_importance.csv"
    roc_path = MODELS_DIR / "best_model_roc.npz"
    metrics_json_path = MODELS_DIR / "metrics.json"

    missing_files = [
        path
        for path in [model_path, scaler_path, imputer_path, metadata_path, metrics_path, importance_path, roc_path, metrics_json_path]
        if not path.exists()
    ]
    if missing_files:
        st.error("Missing trained artifacts. Please run `python -m src.train` first.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metrics_df = pd.read_csv(metrics_path)
    importance_df = pd.read_csv(importance_path)
    roc_data = np.load(roc_path)
    
    with open(metrics_json_path, "r", encoding="utf-8") as file:
        metrics_json = json.load(file)

    return model, scaler, imputer, metadata, metrics_df, importance_df, roc_data, metrics_json


def inject_custom_style():
    """Apply custom CSS to improve app styling."""
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(135deg, #f7f8ea 0%, #e5f6f5 100%);
            }
            .stButton > button {
                border-radius: 10px;
                border: 1px solid #0f766e;
                color: #0f766e;
                font-weight: 600;
            }
            .prediction-card {
                padding: 1rem;
                border-radius: 12px;
                background: #ffffff;
                border-left: 6px solid #0f766e;
                box-shadow: 0 3px 12px rgba(0,0,0,0.08);
                color: #000000;
            }
            .prediction-card h4 {
                color: #000000;
            }
            .prediction-card p {
                color: #000000;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_default_inputs():
    """Default values for the prediction form."""
    return {
        "age": 50,
        "bmi": 28.0,
        "glucose": 6.5,
        "cholesterol": 200.0,
        "systolic_bp": 120.0,
        "physical_activity_level": 4.5,
    }


def initialize_state():
    """Initialize session state values for form fields."""
    defaults = get_default_inputs()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_inputs_callback():
    """Reset form values to defaults via callback."""
    defaults = get_default_inputs()
    for key, value in defaults.items():
        st.session_state[key] = value


def validate_input(values: dict) -> tuple[bool, str]:
    """Validate user inputs before prediction."""
    if not (1 <= values["age"] <= 120):
        return False, "Age must be between 1 and 120."
    if not (10.0 <= values["bmi"] <= 80.0):
        return False, "BMI must be between 10 and 80."
    if not (4.0 <= values["glucose"] <= 20.0):
        return False, "Glucose must be between 4.0 and 20.0."
    if not (100.0 <= values["cholesterol"] <= 450.0):
        return False, "Cholesterol must be between 100 and 450 mg/dL."
    if not (40.0 <= values["systolic_bp"] <= 220.0):
        return False, "Systolic Blood Pressure must be between 40 and 220."
    if not (0.0 <= values["physical_activity_level"] <= 7.0):
        return False, "Physical Activity Level must be between 0.0 and 7.0."
    return True, ""


def get_risk_level(probability: float) -> str:
    """Map probability to human-readable risk levels."""
    if probability < 0.35:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def make_prediction(model, scaler, imputer, feature_order, values: dict):
    """Generate prediction label and probability from model."""
    input_df = pd.DataFrame([values])[feature_order]
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    probability = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(model.predict(input_scaled)[0])
    label = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return label, probability, get_risk_level(probability)


def plot_feature_importance(importance_df: pd.DataFrame):
    """Render horizontal feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_df = importance_df.sort_values("importance", ascending=True)
    ax.barh(sorted_df["feature"], sorted_df["importance"], color="#0f766e")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)


def plot_roc_curve(roc_data):
    """Render ROC curve for the best model."""
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#ef4444", linewidth=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Baseline")
    ax.set_title("Best Model ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)


def plot_confusion_matrix(metrics_json_data):
    """Render confusion matrix for the best model."""
    if "details" not in metrics_json_data:
        return
    
    best_model = metrics_json_data.get("best_model", "Logistic Regression")
    cm = metrics_json_data["details"][best_model]["confusion_matrix"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Diabetic", "Diabetic"])
    ax.set_yticklabels(["Non-Diabetic", "Diabetic"])
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix - {best_model}")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > max(cm[i]) / 2 else "black", fontsize=14, fontweight="bold")
    
    st.pyplot(fig)


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="AI Diabetes Prediction", page_icon="🩺", layout="wide")
    inject_custom_style()
    initialize_state()

    model, scaler, imputer, metadata, metrics_df, importance_df, roc_data, metrics_json = load_artifacts()
    feature_order = metadata.get("selected_features", ["age", "bmi", "glucose", "bp", "cholesterol"])

    st.title("AI-Based Diabetes Prediction System")
    st.caption("Predict diabetes risk using a tuned machine learning model.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Inputs")
        st.number_input("Age (years)", min_value=1, max_value=120, step=1, key="age")
        st.number_input("BMI", min_value=10.0, max_value=80.0, step=0.1, key="bmi")
        st.number_input("Glucose Level", min_value=4.0, max_value=20.0, step=0.1, key="glucose")
        st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=450.0, step=1.0, key="cholesterol")
        st.number_input("Systolic Blood Pressure (mmHg)", min_value=40.0, max_value=220.0, step=1.0, key="systolic_bp")
        st.number_input("Physical Activity Level (0-7)", min_value=0.0, max_value=7.0, step=0.1, key="physical_activity_level")

        predict_col, reset_col = st.columns(2)
        with predict_col:
            predict_clicked = st.button("Predict")
        with reset_col:
            st.button("Reset", on_click=reset_inputs_callback)

        if predict_clicked:
            user_values = {
                "age": float(st.session_state.age),
                "bmi": float(st.session_state.bmi),
                "glucose": float(st.session_state.glucose),
                "cholesterol": float(st.session_state.cholesterol),
                "systolic_bp": float(st.session_state.systolic_bp),
                "physical_activity_level": float(st.session_state.physical_activity_level),
            }

            valid, validation_message = validate_input(user_values)
            if not valid:
                st.error(validation_message)
            else:
                label, probability, risk_level = make_prediction(
                    model,
                    scaler,
                    imputer,
                    feature_order,
                    user_values,
                )
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h4>Prediction: {label}</h4>
                        <p><strong>Probability Score:</strong> {probability:.2%}</p>
                        <p><strong>Risk Level:</strong> {risk_level}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col2:
        st.subheader("Model Performance")
        st.dataframe(metrics_df, width='stretch')
        plot_feature_importance(importance_df)
        st.divider()
        col2_a, col2_b = st.columns(2)
        with col2_a:
            plot_confusion_matrix(metrics_json)
        with col2_b:
            plot_roc_curve(roc_data)

        heatmap_path = MODELS_DIR / "correlation_heatmap.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), caption="Feature Correlation Heatmap")


if __name__ == "__main__":
    main()
