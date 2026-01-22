from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .scoring import compute_thi

MODEL_PATH = "app/ml_model/milk_model.joblib"

FEATURE_COLS = [
    "dim",
    "breed",
    "parity",
    "weight_kg",
    "feed_intake_kg",
    "body_temp_c",
    "rumination_min",
    "eating_min",
    "activity_index",
    "ambient_temp_c",
    "humidity_pct",
    "thi",
]
TARGET_COL = "milk_liters"

def _build_pipeline() -> Pipeline:
    cat_cols = ["breed"]
    num_cols = [c for c in FEATURE_COLS if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )

    return Pipeline([("pre", pre), ("model", model)])

def _prep_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    if df.empty:
        raise ValueError("No rows provided.")

    df = df.dropna(subset=[TARGET_COL])

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = None

    for col in df.columns:
        if col == "breed":
            df[col] = df[col].fillna("Unknown")
        elif col in FEATURE_COLS or col == TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in FEATURE_COLS:
        if col != "breed":
            med = df[col].median()
            df[col] = df[col].fillna(med)

    return df

def train_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = _prep_df(rows)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    pipe = _build_pipeline()
    pipe.fit(X, y)

    joblib.dump(pipe, MODEL_PATH)

    return {
        "trained": True,
        "rows_used": int(len(df)),
        "model_path": MODEL_PATH,
        "features": FEATURE_COLS,
    }

def evaluate_from_rows(rows: List[Dict[str, Any]], test_size: float = 0.2) -> Dict[str, Any]:
    df = _prep_df(rows)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds, squared=False))

    return {
        "rows_used": int(len(df)),
        "test_size": test_size,
        "mae_liters": round(mae, 3),
        "rmse_liters": round(rmse, 3),
    }

def load_model() -> Optional[Pipeline]:
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def build_feature_row(
    *,
    dim: int,
    breed: str,
    parity: Optional[int],
    weight_kg: Optional[float],
    feed_intake_kg: Optional[float],
    body_temp_c: Optional[float],
    rumination_min: Optional[float],
    eating_min: Optional[float],
    activity_index: Optional[float],
    ambient_temp_c: Optional[float],
    humidity_pct: Optional[float],
) -> Dict[str, Any]:
    thi = compute_thi(ambient_temp_c, humidity_pct)
    return {
        "dim": dim,
        "breed": breed,
        "parity": parity,
        "weight_kg": weight_kg,
        "feed_intake_kg": feed_intake_kg,
        "body_temp_c": body_temp_c,
        "rumination_min": rumination_min,
        "eating_min": eating_min,
        "activity_index": activity_index,
        "ambient_temp_c": ambient_temp_c,
        "humidity_pct": humidity_pct,
        "thi": thi,
    }

def forecast_next_days(
    *,
    model: Pipeline,
    start_date: date,
    days: int,
    base_features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    preds = []
    for i in range(days):
        row = dict(base_features)
        row["dim"] = int(base_features["dim"]) + i + 1
        X = pd.DataFrame([row])[FEATURE_COLS]
        yhat = float(model.predict(X)[0])
        preds.append({
            "date": (start_date + timedelta(days=i+1)).isoformat(),
            "predicted_milk_liters": round(max(0.0, yhat), 2),
        })
    return preds
