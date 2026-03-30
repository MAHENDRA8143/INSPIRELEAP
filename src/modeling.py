from __future__ import annotations

from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def train_and_tune_models(x_train, y_train) -> Tuple[Dict[str, object], Dict[str, dict]]:
    trained_models: Dict[str, object] = {}
    best_params: Dict[str, dict] = {}

    logistic = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logistic_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    logistic_search = GridSearchCV(
        estimator=logistic,
        param_grid=logistic_grid,
        scoring="roc_auc",
        cv=2,
        n_jobs=1,
    )
    logistic_search.fit(x_train, y_train)
    trained_models["Logistic Regression"] = logistic_search.best_estimator_
    best_params["Logistic Regression"] = logistic_search.best_params_

    random_forest = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = {
        "n_estimators": [80, 150],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }
    rf_search = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=rf_grid,
        n_iter=2,
        scoring="roc_auc",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf_search.fit(x_train, y_train)
    trained_models["Random Forest"] = rf_search.best_estimator_
    best_params["Random Forest"] = rf_search.best_params_

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        objective="binary:logistic",
    )
    xgb_grid = {
        "n_estimators": [80, 150],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.2],
        "subsample": [0.85, 1.0],
        "colsample_bytree": [0.85, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_grid,
        n_iter=2,
        scoring="roc_auc",
        cv=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_search.fit(x_train, y_train)
    trained_models["XGBoost"] = xgb_search.best_estimator_
    best_params["XGBoost"] = xgb_search.best_params_

    return trained_models, best_params
