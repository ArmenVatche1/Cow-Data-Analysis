from __future__ import annotations

import random
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy.orm import Session

from app.db import Base, engine, SessionLocal
from app.models import Cow, DailyRecord, ExpectedMilkBaseline

random.seed(42)

def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def seed_baseline(db: Session):
    rows = [
        ("Holstein", 0, 30, 28.0, 3.0),
        ("Holstein", 31, 60, 34.0, 3.5),
        ("Holstein", 61, 120, 32.0, 3.5),
        ("Holstein", 121, 200, 28.0, 3.0),
        ("Holstein", 201, 305, 22.0, 2.5),

        ("Jersey", 0, 30, 18.0, 2.3),
        ("Jersey", 31, 60, 22.0, 2.6),
        ("Jersey", 61, 120, 21.0, 2.6),
        ("Jersey", 121, 200, 18.5, 2.3),
        ("Jersey", 201, 305, 15.0, 2.0),

        ("ALL", 0, 30, 24.0, 3.0),
        ("ALL", 31, 60, 28.0, 3.5),
        ("ALL", 61, 120, 27.0, 3.5),
        ("ALL", 121, 200, 24.0, 3.0),
        ("ALL", 201, 305, 19.0, 2.5),
    ]
    for breed, dmin, dmax, exp, sd in rows:
        db.add(ExpectedMilkBaseline(breed=breed, dim_min=dmin, dim_max=dmax, expected_liters=exp, std_dev=sd))

def pick_expected(db: Session, breed: str, dim: int) -> float:
    row = (
        db.query(ExpectedMilkBaseline)
        .filter(ExpectedMilkBaseline.breed == breed)
        .filter(ExpectedMilkBaseline.dim_min <= dim)
        .filter(ExpectedMilkBaseline.dim_max >= dim)
        .first()
    )
    if row:
        return float(row.expected_liters)

    row = (
        db.query(ExpectedMilkBaseline)
        .filter(ExpectedMilkBaseline.breed == "ALL")
        .filter(ExpectedMilkBaseline.dim_min <= dim)
        .filter(ExpectedMilkBaseline.dim_max >= dim)
        .first()
    )
    return float(row.expected_liters) if row else 0.0

def add_record(
    db: Session,
    cow_id: str,
    rec_date: date,
    milk: float,
    feed: float,
    temp: float,
    rum: float,
    eat: float,
    act: float,
    amb: float,
    hum: float,
):
    db.add(DailyRecord(
        cow_id=cow_id,
        date=rec_date,
        milk_liters=round(max(0.0, milk), 2),
        feed_intake_kg=round(max(0.0, feed), 2),
        body_temp_c=round(temp, 2),
        rumination_min=round(max(0.0, rum), 1),
        eating_min=round(max(0.0, eat), 1),
        activity_index=round(max(0.0, act), 1),
        ambient_temp_c=round(amb, 1),
        humidity_pct=round(min(100.0, max(0.0, hum)), 1),
    ))

def seed_scenarios(db: Session, days: int = 30):
    today = date.today()
    start = today - timedelta(days=days-1)

    # Fixed demo cows (IDs you can reference during presentations)
    demo_cows = [
        # A) Healthy high producer
        dict(cow_id="DEMO-A-HEALTHY", breed="Holstein", last_calving_days_ago=60, parity=2, weight=650),
        # B) Heat stress -> THI spikes -> milk drops
        dict(cow_id="DEMO-B-HEAT", breed="Holstein", last_calving_days_ago=45, parity=3, weight=640),
        # C) Fever -> feed/rumination drop first -> then milk drop
        dict(cow_id="DEMO-C-FEVER", breed="Holstein", last_calving_days_ago=80, parity=2, weight=670),
        # D) Appetite/feed issue -> milk lower
        dict(cow_id="DEMO-D-FEED", breed="Jersey", last_calving_days_ago=55, parity=2, weight=460),
        # E) Mild warning (slightly low)
        dict(cow_id="DEMO-E-WARN", breed="Holstein", last_calving_days_ago=110, parity=4, weight=660),
    ]

    for c in demo_cows:
        cow = Cow(
            cow_id=c["cow_id"],
            breed=c["breed"],
            birth_date=today - relativedelta(years=random.randint(3, 6), months=random.randint(0, 11)),
            last_calving_date=today - timedelta(days=c["last_calving_days_ago"]),
            parity=c["parity"],
            weight_kg=float(c["weight"]),
        )
        db.add(cow)
    db.commit()

    # Scenario curves
    for d in range(days):
        rec_date = start + timedelta(days=d)
        for c in demo_cows:
            cow_id = c["cow_id"]
            breed = c["breed"]
            # DIM at this day
            last_calving_date = today - timedelta(days=c["last_calving_days_ago"])
            dim = max(0, (rec_date - last_calving_date).days)

            expected = pick_expected(db, breed, dim)

            # default "healthy-ish" signals
            amb = 18.0 + random.gauss(0, 1.5)
            hum = 60.0 + random.gauss(0, 6.0)
            feed = 21.0 + random.gauss(0, 1.2)
            rum = 430.0 + random.gauss(0, 16.0)
            eat = 300.0 + random.gauss(0, 12.0)
            act = 55.0 + random.gauss(0, 6.0)
            temp = 38.6 + random.gauss(0, 0.08)

            milk = expected + random.gauss(0, 1.2)

            # Apply scenario changes
            if cow_id == "DEMO-A-HEALTHY":
                milk *= 1.05
                feed *= 1.02

            elif cow_id == "DEMO-B-HEAT":
                # Heat wave last 8 days
                if d >= days - 8:
                    amb += 10.5
                    hum += 12.0
                    rum *= 0.88
                    feed *= 0.92
                    milk *= 0.86
                    act *= 1.12

            elif cow_id == "DEMO-C-FEVER":
                # Fever episode: first feed/rum drops, then milk drops
                if d >= days - 9 and d <= days - 6:
                    temp += 0.9
                    feed *= 0.86
                    rum *= 0.86
                if d >= days - 6:
                    milk *= 0.82

            elif cow_id == "DEMO-D-FEED":
                # Appetite issue throughout: lower feed + rumination, milk moderately down
                feed *= 0.82
                rum *= 0.90
                milk *= 0.88

            elif cow_id == "DEMO-E-WARN":
                # Slightly low milk
                milk *= 0.92

            add_record(db, cow_id, rec_date, milk, feed, temp, rum, eat, act, amb, hum)

    db.commit()

def seed_random_herd(db: Session, n_cows: int = 35, days: int = 30):
    today = date.today()
    start = today - timedelta(days=days-1)
    breeds = ["Holstein", "Jersey", "Holstein", "Holstein"]

    # Create cows
    for i in range(n_cows):
        cow_id = f"COW-{2000+i}"
        breed = random.choice(breeds)
        birth_date = today - relativedelta(years=random.randint(2, 7), months=random.randint(0, 11))
        last_calving_date = today - timedelta(days=random.randint(10, 240))
        parity = random.randint(1, 5)
        weight_kg = random.uniform(420, 720) if breed == "Holstein" else random.uniform(360, 520)

        db.add(Cow(
            cow_id=cow_id,
            breed=breed,
            birth_date=birth_date,
            last_calving_date=last_calving_date,
            parity=parity,
            weight_kg=round(weight_kg, 1),
        ))
    db.commit()

    cows = db.query(Cow).filter(Cow.cow_id.like("COW-2%")).all()
    underperformers = set(random.sample([c.cow_id for c in cows], k=max(6, n_cows // 6)))

    base_amb = random.uniform(10.0, 22.0)
    base_hum = random.uniform(45.0, 70.0)

    for cow in cows:
        for d in range(days):
            rec_date = start + timedelta(days=d)
            dim = max(0, (rec_date - cow.last_calving_date).days)
            expected = pick_expected(db, cow.breed, dim)

            milk = max(0.0, expected + random.gauss(0, 1.8))

            amb = base_amb + random.gauss(0, 2.0)
            hum = max(20.0, min(95.0, base_hum + random.gauss(0, 6.0)))

            feed = max(0.0, random.uniform(18.0, 26.0) + (milk - expected) * 0.15 + random.gauss(0, 1.2))
            rum = max(0.0, random.uniform(380, 520) + random.gauss(0, 18))
            eat = max(0.0, random.uniform(240, 340) + random.gauss(0, 14))
            act = max(0.0, random.uniform(35, 85) + random.gauss(0, 6))
            temp = random.uniform(38.0, 39.2) + random.gauss(0, 0.08)

            if cow.cow_id in underperformers:
                milk *= random.uniform(0.62, 0.82)
                feed *= random.uniform(0.72, 0.92)
                rum *= random.uniform(0.75, 0.92)
                if random.random() < 0.35:
                    temp += random.uniform(0.4, 1.1)

            if random.random() < 0.10:
                amb += random.uniform(6.0, 12.0)
                hum = min(95.0, hum + random.uniform(5.0, 15.0))
                rum *= random.uniform(0.82, 0.92)
                act *= random.uniform(1.05, 1.25)

            add_record(db, cow.cow_id, rec_date, milk, feed, temp, rum, eat, act, amb, hum)

    db.commit()

def main():
    reset_db()
    db = SessionLocal()
    try:
        seed_baseline(db)
        seed_scenarios(db, days=30)
        seed_random_herd(db, n_cows=35, days=30)
        db.commit()
        print("✅ Seed complete: scenario-based demo cows + random herd created in cow_milk.db")
        print("Demo cow IDs:")
        print("  DEMO-A-HEALTHY, DEMO-B-HEAT, DEMO-C-FEVER, DEMO-D-FEED, DEMO-E-WARN")
    finally:
        db.close()

if __name__ == "__main__":
    main()
