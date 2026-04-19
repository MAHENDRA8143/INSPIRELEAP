# Diabetes Prediction Project Documentation

## 1. Project Overview
This project is an end-to-end machine learning system for diabetes risk prediction.
It includes:
- Data ingestion from CSV files or fallback dataset
- Data preprocessing and feature preparation
- Training and tuning of multiple classification models
- Evaluation and model selection
- Artifact export for deployment and Streamlit app inference

Primary code modules are in src, model artifacts are in models, and raw/processed datasets are in data.

## 2. Dataset Description

### 2.1 Data Sources
The pipeline supports two input paths:
- Primary path: data/Diabetes_Final_Data_Augmented.csv
- Optional user-specified CSV via command line argument --csv-path
- Fallback path: sklearn diabetes dataset (442 rows), converted to binary classification

Files present in this project:
- data/Diabetes_Final_Data_V2.csv
- data/Diabetes_Final_Data_Augmented.csv
- data/diabetes_dataset.csv

### 2.2 Dataset Sizes and Columns
1) data/Diabetes_Final_Data_V2.csv
- Rows: 5437
- Columns: age, gender, pulse_rate, systolic_bp, diastolic_bp, glucose, height, weight, bmi, family_diabetes, hypertensive, family_hypertension, cardiovascular_disease, stroke, diabetic

2) data/Diabetes_Final_Data_Augmented.csv
- Rows: 5437
- Columns: age, gender, pulse_rate, systolic_bp, diastolic_bp, glucose, height, weight, bmi, family_diabetes, hypertensive, family_hypertension, cardiovascular_disease, stroke, diabetic, cholesterol, physical_activity_level

3) data/diabetes_dataset.csv
- Rows: 442
- Columns: age, sex, bmi, bp, cholesterol, s2, s3, s4, s5, glucose, diabetes

### 2.3 Target Variable
The training target is diabetes (binary: 0 or 1).
- If a dataset contains diabetic with values Yes/No, it is converted to diabetes:
  - Yes -> 1
  - No -> 0
- If using sklearn fallback, continuous target is converted to binary using median threshold.

### 2.4 Feature Set Used for Modeling
The selected model input features are:
- age
- bmi
- glucose
- cholesterol
- systolic_bp
- physical_activity_level

These are configured in src/config.py as MODEL_FEATURES.

### 2.5 Augmentation Logic
scripts/augment_dataset.py creates two synthetic but clinically motivated columns from Diabetes_Final_Data_V2.csv:
- cholesterol: derived from age, bmi, glucose plus Gaussian noise, clipped to [100, 450]
- physical_activity_level: derived inversely from bmi and age plus noise, clipped to [0, 7]

This generates data/Diabetes_Final_Data_Augmented.csv used by default if available.

## 3. Preprocessing Techniques
The preprocessing pipeline is implemented in src/preprocessing.py and used by src/train.py.

### 3.1 Train/Test Split
- Method: stratified split
- Ratio: 80% train, 20% test
- Test size: 0.2
- Random seed: 42

### 3.2 Missing Value Handling
- Method: SimpleImputer with median strategy
- Fit on training data only
- Applied to both training and test sets

### 3.3 Feature Scaling
- Method: StandardScaler
- Fit on imputed training data
- Applied to both train and test sets

### 3.4 Class Imbalance Handling
- Method: SMOTE oversampling
- Condition: applied only when minority/majority ratio < 0.5
- Current trained artifacts indicate used_smote = true

### 3.5 Feature Engineering Utilities
In src/feature_engineering.py:
- Correlation heatmap generation for numeric features
- Mutual information ranking via mutual_info_classif
- Explicit selection of MODEL_FEATURES

Saved outputs:
- models/correlation_heatmap.png
- models/feature_ranking.csv
- models/feature_importance.csv

## 4. Model Training and Results

### 4.1 Models Trained
Implemented in src/modeling.py:
- Logistic Regression (GridSearchCV)
- Random Forest (RandomizedSearchCV)
- XGBoost (RandomizedSearchCV)

All models are tuned using ROC-AUC as scoring metric with 2-fold cross-validation.

### 4.2 Best Hyperparameters from Trained Artifacts
From models/model_metadata.json:
- Logistic Regression: C=1, penalty=l2, solver=lbfgs
- Random Forest: n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight=None
- XGBoost: n_estimators=150, max_depth=5, learning_rate=0.2, subsample=1.0, colsample_bytree=0.85

### 4.3 Model Selection
Best model is selected by highest ROC-AUC on test set.
Current best model: Logistic Regression.

### 4.4 Saved Artifacts
Training exports the following production artifacts:
- models/best_model.joblib
- models/scaler.joblib
- models/imputer.joblib
- models/model_metadata.json
- models/metrics.csv
- models/metrics.json
- models/best_model_roc.npz
- models/feature_ranking.csv
- models/feature_importance.csv

## 5. Evaluation Metrics
Evaluation logic is in src/evaluate.py.
Metrics computed for each model:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve arrays (FPR, TPR, thresholds)

### 5.1 Formula Summary
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- ROC-AUC: Area under ROC curve, where ROC plots TPR against FPR at varying thresholds

### 5.2 Recorded Test Metrics
From models/metrics.csv:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7564 | 0.1597 | 0.6667 | 0.2577 | 0.7957 |
| Random Forest | 0.8539 | 0.2289 | 0.5507 | 0.3234 | 0.7768 |
| XGBoost | 0.9210 | 0.3607 | 0.3188 | 0.3385 | 0.7743 |

### 5.3 Best Model Confusion Matrix
For Logistic Regression (from models/metrics.json):
- TN: 777
- FP: 242
- FN: 23
- TP: 46

Interpretation:
- Logistic Regression has the best ROC-AUC in this run.
- It also has the highest recall among compared models, which can be valuable in screening scenarios where missing positive cases is costly.
- XGBoost shows the highest accuracy, indicating stronger overall correctness but lower recall compared to Logistic Regression.

## 6. End-to-End Training Flow
1. Load dataset (user CSV, augmented CSV, or sklearn fallback).
2. Build binary target diabetes.
3. Select fixed feature subset.
4. Split train/test with stratification.
5. Impute missing values with training median.
6. Scale features using StandardScaler.
7. Apply SMOTE if imbalance threshold is met.
8. Train and tune Logistic Regression, Random Forest, and XGBoost.
9. Evaluate all models on holdout test set.
10. Select best model by ROC-AUC and save artifacts.

## 7. Reproducibility and Configuration
Key constants in src/config.py:
- RANDOM_STATE = 42
- TEST_SIZE = 0.2
- TARGET_COLUMN = diabetes
- MODEL_FEATURES = [age, bmi, glucose, cholesterol, systolic_bp, physical_activity_level]

Dependencies are declared in requirements.txt, including:
- pandas, numpy, scikit-learn
- imbalanced-learn, xgboost
- matplotlib, seaborn
- joblib, streamlit

## 8. How to Run
- Train with default data path logic:
  python -m src.train

- Train with specific CSV file:
  python -m src.train --csv-path data/Diabetes_Final_Data_V2.csv

- Run app:
  streamlit run app/app.py

## 9. Notes
- If the input CSV does not contain diabetes but contains diabetic, automatic conversion is performed.
- If neither custom CSV nor augmented CSV exists, sklearn dataset fallback is used.
- Artifact files in models represent the latest completed training run.
