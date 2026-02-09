import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ARTIFACTS_DIR = os.path.join("artifacts")
DATA_DIR = os.path.join("..", "data")
TEST_SAMPLE_PATH = os.path.join(DATA_DIR, "test_sample.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "feature_names.json")
METRICS_CSV = os.path.join(ARTIFACTS_DIR, "metrics_summary.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def _fit_scaler(X_train, feature_names):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train[feature_names])
    return scaler, Xs

def _transform_with_optional_scaler(scaler, X, feature_names):
    if scaler is None:
        return X[feature_names]
    return scaler.transform(X[feature_names])

def load_dataset():
    csv_path = os.path.join(DATA_DIR, "fetal_health.csv")
    target_col = "fetal_health"

    df = pd.read_csv(csv_path)

    feature_names = [c for c in df.columns if c != target_col]
    X: pd.DataFrame = df[feature_names].copy()
    y: pd.Series = df[target_col].copy()
    return X, y, feature_names

def train_logistic_regression(X_train, y_train, feature_names):
    scaler, Xs = _fit_scaler(X_train, feature_names)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(Xs, y_train)
    return clf, scaler

def train_knn(X_train, y_train, feature_names, n_neighbors = 5):
    scaler, Xs = _fit_scaler(X_train, feature_names)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(Xs, y_train)
    return clf, scaler

def train_gaussian_nb(X_train, y_train, feature_names):
    scaler, Xs = _fit_scaler(X_train, feature_names)
    clf = GaussianNB()
    clf.fit(Xs, y_train)
    return clf, scaler

def train_decision_tree(X_train, y_train, feature_names, random_state = 42):
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train[feature_names], y_train)
    return clf, None

def train_random_forest(X_train, y_train, feature_names, n_estimators = 200, random_state = 42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train[feature_names], y_train)
    return clf, None

def train_xgboost(X_train, y_train, feature_names, n_classes, random_state = 42):
    y0 = pd.Series(y_train).astype(int) - 1

    params = dict(
        n_estimators=300, learning_rate=0.1, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=random_state, n_jobs=4,
        objective="multi:softprob", num_class=n_classes,
    )
    clf = XGBClassifier(**params)
    clf.fit(X_train[feature_names], y0)
    return clf, None

def predict(estimator, scaler, X, feature_names):
    X_in = _transform_with_optional_scaler(scaler, X, feature_names)
    y_pred = estimator.predict(X_in)

    if estimator.__class__.__name__ == "XGBClassifier":
        return y_pred + 1
    return y_pred

def predict_proba(estimator, scaler, X, feature_names):
    if not hasattr(estimator, "predict_proba"):
        return None
    X_in = _transform_with_optional_scaler(scaler, X, feature_names)
    try:
        return estimator.predict_proba(X_in)
    except Exception:
        return None

def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["Precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    out["F1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                out["AUC"] = float(roc_auc_score(y_true, y_proba))
            else:
                out["AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            out["AUC"] = float("nan")
    else:
        out["AUC"] = float("nan")
    return out

def main():
    X, y, feature_names = load_dataset()

    X_train, X_test, y_train, y_test, y_train, y_test = train_test_split(
        X, y, y, test_size=0.2, stratify=y, random_state=42
    )

    test_df = X_test.copy()
    test_df["fetal_health"] = y_test.values
    test_df.to_csv(TEST_SAMPLE_PATH, index=False)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "target_name": "fetal_health"}, f, indent=2)

    n_classes = int(pd.Series(y_train).nunique())

    models: Dict[str, Tuple[object, Optional[StandardScaler]]] = {}
    models["LogisticRegression"] = train_logistic_regression(X_train, y_train, feature_names)
    models["DecisionTree"] = train_decision_tree(X_train, y_train, feature_names)
    models["KNN"] = train_knn(X_train, y_train, feature_names)
    models["GaussianNB"] = train_gaussian_nb(X_train, y_train, feature_names)
    models["RandomForest"] = train_random_forest(X_train, y_train, feature_names)
    models["XGBoost"] = train_xgboost(X_train, y_train, feature_names, n_classes)

    for name, (est, scaler) in models.items():
        out_path = os.path.join(ARTIFACTS_DIR, f"{name}.joblib")
        joblib.dump({"estimator": est, "scaler": scaler, "feature_names": feature_names}, out_path)

    results = []
    for name, (est, scaler) in models.items():
        y_pred = predict(est, scaler, X_test, feature_names)
        y_proba = predict_proba(est, scaler, X_test, feature_names)
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    metrics_df.rename(columns={"Model": "ML Model Name"}, inplace=True)
    metrics_df = metrics_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics_df.sort_values(by="Accuracy", ascending=False, inplace=True)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\nTraining complete. Metrics saved to:", METRICS_CSV)
    print("Artifacts directory:", ARTIFACTS_DIR)
    print("Sample test CSV:", TEST_SAMPLE_PATH)

if __name__ == "__main__":
    main()
