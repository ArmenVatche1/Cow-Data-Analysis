from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .db import Base, engine, get_db
from .models import Cow, DailyRecord, ExpectedMilkBaseline, ScanEvent
from .schemas import CowCreate, DailyRecordCreate
from .services.baseline import BaselineRow, BaselineTable
from .services.scoring import score_cow
from .services.ml import train_from_rows, evaluate_from_rows, load_model, build_feature_row, forecast_next_days

app = FastAPI(title="Cow Milk Monitor", version="0.4.0")
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

@app.get("/")
def root():
    return {"service": "Cow Milk Monitor API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True}

# ---------------------------
# Cows
# ---------------------------
@app.get("/cows")
def list_cows(db: Session = Depends(get_db)):
    cows = db.query(Cow).order_by(Cow.cow_id).all()
    return [
        {
            "cow_id": c.cow_id,
            "breed": c.breed,
            "birth_date": c.birth_date.isoformat(),
            "last_calving_date": c.last_calving_date.isoformat(),
            "parity": c.parity,
            "weight_kg": c.weight_kg,
        }
        for c in cows
    ]

@app.post("/cows")
def create_cow(payload: CowCreate, db: Session = Depends(get_db)):
    existing = db.query(Cow).filter(Cow.cow_id == payload.cow_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="cow_id already exists.")

    cow = Cow(
        cow_id=payload.cow_id,
        breed=payload.breed,
        birth_date=payload.birth_date,
        last_calving_date=payload.last_calving_date,
        parity=payload.parity,
        weight_kg=payload.weight_kg,
    )
    db.add(cow)
    db.commit()
    return {"created": True, "cow_id": cow.cow_id}

# ---------------------------
# Records
# ---------------------------
@app.get("/cows/{cow_id}/records")
def cow_records(cow_id: str, days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found.")

    recs = (
        db.query(DailyRecord)
        .filter(DailyRecord.cow_id == cow_id)
        .order_by(DailyRecord.date)
        .all()
    )
    recs = recs[-days:] if len(recs) > days else recs

    return [
        {
            "date": r.date.isoformat(),
            "milk_liters": r.milk_liters,
            "feed_intake_kg": r.feed_intake_kg,
            "body_temp_c": r.body_temp_c,
            "rumination_min": r.rumination_min,
            "eating_min": r.eating_min,
            "activity_index": r.activity_index,
            "ambient_temp_c": r.ambient_temp_c,
            "humidity_pct": r.humidity_pct,
        }
        for r in recs
    ]

@app.post("/records")
def create_record(payload: DailyRecordCreate, db: Session = Depends(get_db)):
    cow = db.query(Cow).filter(Cow.cow_id == payload.cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found. Create cow first.")

    existing = (
        db.query(DailyRecord)
        .filter(DailyRecord.cow_id == payload.cow_id)
        .filter(DailyRecord.date == payload.date)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Record for this cow/date already exists.")

    rec = DailyRecord(
        cow_id=payload.cow_id,
        date=payload.date,
        milk_liters=payload.milk_liters,
        feed_intake_kg=payload.feed_intake_kg,
        body_temp_c=payload.body_temp_c,
        rumination_min=payload.rumination_min,
        eating_min=payload.eating_min,
        activity_index=payload.activity_index,
        ambient_temp_c=payload.ambient_temp_c,
        humidity_pct=payload.humidity_pct,
    )
    db.add(rec)
    db.commit()
    return {"created": True, "cow_id": payload.cow_id, "date": payload.date.isoformat()}

# ---------------------------
# Scan + Alerts
# ---------------------------
@app.get("/scan/{cow_id}")
def scan_cow(cow_id: str, include_forecast: bool = Query(default=True), db: Session = Depends(get_db)):
    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found.")

    latest = (
        db.query(DailyRecord)
        .filter(DailyRecord.cow_id == cow_id)
        .order_by(desc(DailyRecord.date))
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No daily record found for this cow.")

    recent = (
        db.query(DailyRecord)
        .filter(DailyRecord.cow_id == cow_id)
        .order_by(DailyRecord.date)
        .all()
    )

    # DIM at latest record date (not today)
    dim = max(0, (latest.date - cow.last_calving_date).days)

    baseline = load_baseline_table(db)
    expected = baseline.lookup(breed=cow.breed, dim=dim)

    scored = score_cow(cow=cow, latest=latest, expected_liters=expected, recent=recent)

    # Optional ML forecast
    forecast = None
    if include_forecast:
        model = load_model()
        if model is not None:
            base = build_feature_row(
                dim=dim,
                breed=cow.breed,
                parity=cow.parity,
                weight_kg=cow.weight_kg,
                feed_intake_kg=latest.feed_intake_kg,
                body_temp_c=latest.body_temp_c,
                rumination_min=latest.rumination_min,
                eating_min=latest.eating_min,
                activity_index=latest.activity_index,
                ambient_temp_c=latest.ambient_temp_c,
                humidity_pct=latest.humidity_pct,
            )
            forecast = forecast_next_days(model=model, start_date=latest.date, days=7, base_features=base)

    scored["ml_forecast_7d"] = forecast

    # Audit log scan
    db.add(
        ScanEvent(
            cow_id=cow.cow_id,
            timestamp=datetime.utcnow(),
            status=scored["status"],
            reasons=",".join(scored.get("reasons", [])),
            risk_score=scored.get("risk_score"),
            expected_liters=scored.get("expected_liters"),
            actual_liters=scored.get("actual_liters"),
            difference_pct=scored.get("difference_pct"),
        )
    )
    db.commit()

    return scored

@app.get("/alerts")
def alerts(
    status: Optional[str] = Query(default=None, description="Filter by OK/WARNING/FLAG/UNKNOWN"),
    db: Session = Depends(get_db),
):
    cows = db.query(Cow).all()
    baseline = load_baseline_table(db)

    results = []
    for cow in cows:
        latest = (
            db.query(DailyRecord)
            .filter(DailyRecord.cow_id == cow.cow_id)
            .order_by(desc(DailyRecord.date))
            .first()
        )
        if not latest:
            continue

        recent = (
            db.query(DailyRecord)
            .filter(DailyRecord.cow_id == cow.cow_id)
            .order_by(DailyRecord.date)
            .all()
        )

        dim = max(0, (latest.date - cow.last_calving_date).days)
        expected = baseline.lookup(breed=cow.breed, dim=dim)
        scored = score_cow(cow=cow, latest=latest, expected_liters=expected, recent=recent)

        if status is None or scored["status"] == status:
            results.append(scored)

    order = {"FLAG": 0, "WARNING": 1, "OK": 2, "UNKNOWN": 3}
    results.sort(key=lambda r: (order.get(r["status"], 9), r["risk_score"] if r.get("risk_score") is not None else 999.0))
    return results

@app.get("/scans/recent")
def recent_scans(limit: int = Query(default=25, ge=1, le=200), db: Session = Depends(get_db)):
    scans = (
        db.query(ScanEvent)
        .order_by(desc(ScanEvent.timestamp))
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp": s.timestamp.isoformat() + "Z",
            "cow_id": s.cow_id,
            "status": s.status,
            "reasons": s.reasons,
            "risk_score": s.risk_score,
            "expected_liters": s.expected_liters,
            "actual_liters": s.actual_liters,
            "difference_pct": s.difference_pct,
        }
        for s in scans
    ]

# ---------------------------
# ML (train + evaluate + predict)
# ---------------------------
def _training_rows(db: Session) -> List[dict]:
    from .services.scoring import compute_thi
    rows: List[dict] = []
    cows = {c.cow_id: c for c in db.query(Cow).all()}

    records = db.query(DailyRecord).order_by(DailyRecord.date).all()
    for r in records:
        cow = cows.get(r.cow_id)
        if not cow:
            continue

        dim = max(0, (r.date - cow.last_calving_date).days)
        thi = compute_thi(r.ambient_temp_c, r.humidity_pct)

        rows.append({
            "dim": dim,
            "breed": cow.breed,
            "parity": cow.parity,
            "weight_kg": cow.weight_kg,
            "feed_intake_kg": r.feed_intake_kg,
            "body_temp_c": r.body_temp_c,
            "rumination_min": r.rumination_min,
            "eating_min": r.eating_min,
            "activity_index": r.activity_index,
            "ambient_temp_c": r.ambient_temp_c,
            "humidity_pct": r.humidity_pct,
            "thi": thi,
            "milk_liters": r.milk_liters,
        })
    return rows

@app.post("/ml/train")
def ml_train(db: Session = Depends(get_db)):
    rows = _training_rows(db)
    if len(rows) < 80:
        raise HTTPException(status_code=400, detail="Not enough rows to train (need ~80+). Reseed with more days/cows.")
    info = train_from_rows(rows)
    return info

@app.get("/ml/evaluate")
def ml_evaluate(test_size: float = Query(default=0.2, ge=0.1, le=0.5), db: Session = Depends(get_db)):
    rows = _training_rows(db)
    if len(rows) < 80:
        raise HTTPException(status_code=400, detail="Not enough rows to evaluate (need ~80+).")
    metrics = evaluate_from_rows(rows, test_size=test_size)
    return metrics

@app.get("/predict/{cow_id}")
def predict(cow_id: str, days: int = Query(default=7, ge=1, le=30), db: Session = Depends(get_db)):
    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found.")

    latest = (
        db.query(DailyRecord)
        .filter(DailyRecord.cow_id == cow_id)
        .order_by(desc(DailyRecord.date))
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No daily record found for this cow.")

    model = load_model()
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call POST /ml/train first.")

    dim = max(0, (latest.date - cow.last_calving_date).days)
    base = build_feature_row(
        dim=dim,
        breed=cow.breed,
        parity=cow.parity,
        weight_kg=cow.weight_kg,
        feed_intake_kg=latest.feed_intake_kg,
        body_temp_c=latest.body_temp_c,
        rumination_min=latest.rumination_min,
        eating_min=latest.eating_min,
        activity_index=latest.activity_index,
        ambient_temp_c=latest.ambient_temp_c,
        humidity_pct=latest.humidity_pct,
    )

    forecast = forecast_next_days(model=model, start_date=latest.date, days=days, base_features=base)
    return {"cow_id": cow_id, "start_date": latest.date.isoformat(), "days": days, "forecast": forecast}
