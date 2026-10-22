"""Training, evaluation and prediction for DataForge AI."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ["brand", "cpu", "gpu", "ram_gb", "storage_gb", "age_years", "condition"]
CATEGORICAL = ["brand", "cpu", "gpu", "condition"]
NUMERIC = ["ram_gb", "storage_gb", "age_years"]

def _build_pipeline(model):
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])

def train_models(df: pd.DataFrame) -> dict:
    clean = df.copy()
    if "price_inr" not in clean:
        raise ValueError("Dataset has no price_inr column.")
    clean["price_inr"] = pd.to_numeric(clean["price_inr"], errors="coerce")
    clean = clean.dropna(subset=["price_inr"])
    clean = clean[(clean.price_inr >= 3000) & (clean.price_inr <= 1_000_000)]
    if len(clean) < 8:
        raise ValueError(f"Need at least 8 valid priced listings to evaluate a prototype; found {len(clean)}.")
    for col in FEATURES:
        if col not in clean:
            clean[col] = np.nan

    X = clean[FEATURES].copy()
    y = clean["price_inr"].astype(float)
    # Hold-out set gives an honest estimate; CV only when there are enough rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    candidates = {
        "Ridge Regression": _build_pipeline(Ridge(alpha=2.0)),
        "Extra Trees": _build_pipeline(ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    }
    reports = {}
    trained = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        reports[name] = {
            "MAE (INR)": round(float(mean_absolute_error(y_test, pred)), 2),
            "RMSE (INR)": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "R²": round(float(r2_score(y_test, pred)), 3) if len(y_test) > 1 else None,
        }
        trained[name] = pipe
        if len(clean) >= 10:
            folds = min(5, len(clean))
            cv = cross_val_score(pipe, X, y, cv=folds, scoring="neg_mean_absolute_error")
            reports[name][f"{folds}-fold CV MAE (INR)"] = round(float(-cv.mean()), 2)

    # Select by holdout MAE; do not assume the complex model always wins.
    best_name = min(reports, key=lambda name: reports[name]["MAE (INR)"])
    # Refit selected model on all collected rows for subsequent predictions.
    best_model = trained[best_name]
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    metadata = {"best_model": best_name, "rows": int(len(clean)), "reports": reports}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    return metadata

def predict_price(inputs: dict) -> float:
    path = MODEL_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Train a model first.")
    model = joblib.load(path)
    row = {field: inputs.get(field) for field in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    prediction = float(model.predict(X)[0])
    return max(0.0, prediction)
