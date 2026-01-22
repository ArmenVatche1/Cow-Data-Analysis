#!/usr/bin/env bash
set -euo pipefail

cd "/Users/armenkevorkian/Desktop/CV projects/Cow-ML-Prototype"

cat > requirements.txt << 'REQ'
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy==2.0.34
pydantic==2.9.2
python-dateutil==2.9.0.post0
pandas==2.2.2
streamlit==1.38.0
scikit-learn==1.5.1
joblib==1.4.2
REQ

pip install -r requirements.txt

mkdir -p app/services models
touch app/__init__.py
touch app/services/__init__.py

cat > app/services/ml.py << 'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from sqlalchemy.orm import Session

from app.models import Cow, DailyRecord, ExpectedMilkBaseline
from app.services.scoring import compute_thi

MODEL_PATH = Path("models/early_warning.joblib")

@dataclass
class TrainResult:
    n_rows: int
    n_train: int
    n_test: int
    metrics: Dict[str, Any]
    feature_cols: List[str]

def _baseline_lookup_rows(db: Session) -> pd.DataFrame:
    rows = db.query(ExpectedMilkBaseline).all()
    return pd.DataFrame([
        {"breed": r.breed, "dim_min": r.dim_min, "dim_max": r.dim_max, "expected_liters": float(r.expected_liters)}
        for r in rows
    ])

def expected_for_dim(baseline_df: pd.DataFrame, breed: str, dim: int) -> float:
    m = baseline_df[
        (baseline_df["breed"] == breed) &
        (baseline_df["dim_min"] <= dim) &
        (baseline_df["dim_max"] >= dim)
    ]
    if not m.empty:
        return float(m.iloc[0]["expected_liters"])

    m = baseline_df[
        (baseline_df["breed"] == "ALL") &
        (baseline_df["dim_min"] <= dim) &
        (baseline_df["dim_max"] >= dim)
    ]
    if not m.empty:
        return float(m.iloc[0]["expected_liters"])

    return 0.0

def _cow_records_df(db: Session, cow_id: str) -> pd.DataFrame:
    recs = (
        db.query(DailyRecord)
        .filter(DailyRecord.cow_id == cow_id)
        .order_by(DailyRecord.date)
        .all()
    )
    if not recs:
        return pd.DataFrame()

    return pd.DataFrame([{
        "date": r.date,
        "milk_liters": r.milk_liters,
        "feed_intake_kg": r.feed_intake_kg,
        "body_temp_c": r.body_temp_c,
        "rumination_min": r.rumination_min,
        "eating_min": r.eating_min,
        "activity_index": r.activity_index,
        "ambient_temp_c": r.ambient_temp_c,
        "humidity_pct": r.humidity_pct,
    } for r in recs]).sort_values("date")

def _flag_rule_for_day(
    *,
    milk: float,
    expected: float,
    feed: Optional[float],
    temp: Optional[float],
    rum: Optional[float],
    thi: Optional[float],
) -> bool:
    if expected > 0 and milk < expected * 0.80:
        return True
    if temp is not None and temp >= 39.5:
        return True
    if rum is not None and rum < 300:
        return True
    if feed is not None and feed < 14.0:
        return True
    if thi is not None and thi >= 80:
        return True
    return False

def _build_features_for_window(window: pd.DataFrame) -> Dict[str, float]:
    w = window.copy()
    w["milk_liters"] = w["milk_liters"].astype(float)

    milk_slope = float(w.iloc[-1]["milk_liters"]) - float(w.iloc[0]["milk_liters"])

    feats: Dict[str, float] = {
        "milk_avg7": float(w["milk_liters"].mean()),
        "milk_min7": float(w["milk_liters"].min()),
        "milk_slope7": float(milk_slope),
    }

    if w["feed_intake_kg"].notna().any():
        feed = w["feed_intake_kg"].astype(float)
        feats["feed_avg7"] = float(feed.mean())
        feats["feed_slope7"] = float(feed.iloc[-1] - feed.iloc[0])
    else:
        feats["feed_avg7"] = 0.0
        feats["feed_slope7"] = 0.0

    if w["body_temp_c"].notna().any():
        temp = w["body_temp_c"].astype(float)
        feats["temp_avg7"] = float(temp.mean())
        feats["temp_max7"] = float(temp.max())
    else:
        feats["temp_avg7"] = 0.0
        feats["temp_max7"] = 0.0

    if w["rumination_min"].notna().any():
        rum = w["rumination_min"].astype(float)
        feats["rum_avg7"] = float(rum.mean())
        feats["rum_slope7"] = float(rum.iloc[-1] - rum.iloc[0])
    else:
        feats["rum_avg7"] = 0.0
        feats["rum_slope7"] = 0.0

    if w["activity_index"].notna().any():
        act = w["activity_index"].astype(float)
        feats["act_avg7"] = float(act.mean())
        feats["act_slope7"] = float(act.iloc[-1] - act.iloc[0])
    else:
        feats["act_avg7"] = 0.0
        feats["act_slope7"] = 0.0

    thi_vals = []
    for _, row in w.iterrows():
        thi = compute_thi(row.get("ambient_temp_c"), row.get("humidity_pct"))
        if thi is not None:
            thi_vals.append(float(thi))
    feats["thi_avg7"] = float(sum(thi_vals) / len(thi_vals)) if thi_vals else 0.0
    feats["thi_max7"] = float(max(thi_vals)) if thi_vals else 0.0

    return feats

def build_dataset(db: Session, *, lookback_days: int = 7, horizon_days: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    cows = db.query(Cow).order_by(Cow.cow_id).all()
    baseline_df = _baseline_lookup_rows(db)

    rows = []
    for cow in cows:
        df = _cow_records_df(db, cow.cow_id)
        if df.empty:
            continue

        df = df.sort_values("date").reset_index(drop=True)

        dim0 = (df.loc[0, "date"] - cow.last_calving_date).days
        expected_list = []
        flag_list = []
        for i in range(len(df)):
            dim = dim0 + i
            expected = expected_for_dim(baseline_df, cow.breed, dim)
            expected_list.append(expected)

            thi = compute_thi(df.loc[i, "ambient_temp_c"], df.loc[i, "humidity_pct"])
            flag = _flag_rule_for_day(
                milk=float(df.loc[i, "milk_liters"]),
                expected=float(expected),
                feed=df.loc[i, "feed_intake_kg"],
                temp=df.loc[i, "body_temp_c"],
                rum=df.loc[i, "rumination_min"],
                thi=thi,
            )
            flag_list.append(bool(flag))

        df["is_flag_day"] = flag_list

        for i in range(lookback_days - 1, len(df) - horizon_days):
            window = df.iloc[i - (lookback_days - 1): i + 1]
            future = df.iloc[i + 1: i + 1 + horizon_days]

            feats = _build_features_for_window(window)
            feats["cow_id"] = cow.cow_id
            feats["breed"] = cow.breed
            feats["parity"] = float(cow.parity or 0)
            feats["weight_kg"] = float(cow.weight_kg or 0)
            feats["dim"] = float((window.iloc[-1]["date"] - cow.last_calving_date).days)

            feats["label_next3d_flag"] = 1 if bool(future["is_flag_day"].any()) else 0
            rows.append(feats)

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset, []

    dataset = pd.get_dummies(dataset, columns=["breed"], drop_first=False)
    feature_cols = [c for c in dataset.columns if c not in ("cow_id", "label_next3d_flag")]
    return dataset, feature_cols

def train_model(db: Session) -> TrainResult:
    ds, feature_cols = build_dataset(db)
    if ds.empty:
        raise ValueError("No dataset rows available. Seed more data first.")

    cow_ids = sorted(ds["cow_id"].unique().tolist())
    split_idx = int(len(cow_ids) * 0.8)
    train_cows = set(cow_ids[:split_idx])
    test_cows = set(cow_ids[split_idx:])

    train_df = ds[ds["cow_id"].isin(train_cows)].copy()
    test_df = ds[ds["cow_id"].isin(test_cows)].copy()

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["label_next3d_flag"].astype(int)

    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["label_next3d_flag"].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1] if len(X_test) else []
    pred = (prob >= 0.5).astype(int) if len(X_test) else []

    if len(X_test):
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = None
    else:
        p = r = f1 = 0.0
        auc = None

    metrics = {
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1": round(float(f1), 3),
        "roc_auc": None if auc is None else round(float(auc), 3),
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
        "positive_rate_train": round(float(y_train.mean()), 3) if len(y_train) else 0.0,
        "positive_rate_test": round(float(y_test.mean()), 3) if len(y_test) else 0.0,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)

    return TrainResult(
        n_rows=int(len(ds)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        metrics=metrics,
        feature_cols=feature_cols,
    )

def load_model_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Call POST /ml/train first.")
    return joblib.load(MODEL_PATH)

def predict_cow_risk(db: Session, cow_id: str, *, lookback_days: int = 7) -> Dict[str, Any]:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise ValueError("Cow not found.")

    df = _cow_records_df(db, cow_id)
    if df.empty or len(df) < lookback_days:
        raise ValueError("Not enough daily records for prediction (need >= 7 days).")

    df = df.sort_values("date")
    window = df.iloc[-lookback_days:].copy()

    feats = _build_features_for_window(window)
    feats["cow_id"] = cow.cow_id
    feats["parity"] = float(cow.parity or 0)
    feats["weight_kg"] = float(cow.weight_kg or 0)
    feats["dim"] = float((window.iloc[-1]["date"] - cow.last_calving_date).days)

    row = {"breed": cow.breed, **feats}
    row_df = pd.DataFrame([row])
    row_df = pd.get_dummies(row_df, columns=["breed"], drop_first=False)

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0.0

    X = row_df[feature_cols].fillna(0.0)
    prob = float(model.predict_proba(X)[:, 1][0])
    risk = "HIGH" if prob >= 0.75 else ("MEDIUM" if prob >= 0.45 else "LOW")

    return {
        "cow_id": cow_id,
        "prob_next3d_flag": round(prob, 3),
        "risk_level": risk,
        "window_end_date": str(window.iloc[-1]["date"]),
    }

def predict_herd_risk(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    cows = db.query(Cow).order_by(Cow.cow_id).all()
    out = []
    for cow in cows:
        try:
            out.append(predict_cow_risk(db, cow.cow_id))
        except Exception:
            continue
    out.sort(key=lambda x: x["prob_next3d_flag"], reverse=True)
    return out[:limit]
PY

cat > app/main.py << 'PY'
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .db import Base, engine, get_db
from .models import Cow, DailyRecord, ExpectedMilkBaseline, ScanEvent
from .services.baseline import BaselineRow, BaselineTable
from .services.scoring import score_cow, days_in_milk
from .services.ml import train_model, predict_cow_risk, predict_herd_risk

app = FastAPI(title="Cow Milk Monitor", version="0.3.0")
Base.metadata.create_all(bind=engine)

def load_baseline_table(db: Session) -> BaselineTable:
    rows = db.query(ExpectedMilkBaseline).all()
    baseline_rows = [
        BaselineRow(
            breed=r.breed,
            dim_min=r.dim_min,
            dim_max=r.dim_max,
            expected_liters=r.expected_liters,
            std_dev=r.std_dev
        )
        for r in rows
    ]
    return BaselineTable(baseline_rows)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ml/train")
def ml_train(db: Session = Depends(get_db)):
    try:
        res = train_model(db)
        return {
            "ok": True,
            "rows": res.n_rows,
            "train_rows": res.n_train,
            "test_rows": res.n_test,
            "metrics": res.metrics,
            "feature_count": len(res.feature_cols),
            "model_path": "models/early_warning.joblib",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ml/predict/{cow_id}")
def ml_predict_cow(cow_id: str, db: Session = Depends(get_db)):
    try:
        return predict_cow_risk(db, cow_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ml/predict/herd")
def ml_predict_herd(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)):
    try:
        return predict_herd_risk(db, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
PY

cat > dashboard/streamlit_app.py << 'PY'
import requests
import pandas as pd
import streamlit as st

API_BASE_DEFAULT = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cow Milk Monitor", layout="wide")
st.title("Cow Milk Monitor (Prototype v3)")
st.caption("Early-Warning ML: predict probability of FLAG in the next 3 days.")

with st.sidebar:
    st.header("API")
    api_base = st.text_input("API Base URL", API_BASE_DEFAULT)
    st.divider()
    st.subheader("ML")
    train_btn = st.button("Train / Re-train Early-Warning Model")

def get_json(url: str, method: str = "GET"):
    if method == "POST":
        r = requests.post(url, timeout=60)
    else:
        r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

if train_btn:
    try:
        res = get_json(f"{api_base}/ml/train", method="POST")
        st.sidebar.success(f"Trained. F1={res['metrics']['f1']} Precision={res['metrics']['precision']} Recall={res['metrics']['recall']}")
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

st.subheader("Herd risk")
try:
    risk = get_json(f"{api_base}/ml/predict/herd?limit=80")
    dfR = pd.DataFrame(risk)
    st.dataframe(dfR, use_container_width=True)
except Exception as e:
    st.error(f"Could not load herd risk: {e}")
PY

echo "Option B files written."
