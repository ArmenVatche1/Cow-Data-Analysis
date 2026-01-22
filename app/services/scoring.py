from __future__ import annotations
from datetime import date
from typing import Dict, Any, List, Optional

def days_in_milk(last_calving_date: date, as_of: date) -> int:
    return max(0, (as_of - last_calving_date).days)

def compute_thi(temp_c: Optional[float], rh_pct: Optional[float]) -> Optional[float]:
    if temp_c is None or rh_pct is None:
        return None
    rh = max(0.0, min(100.0, float(rh_pct)))
    t = float(temp_c)
    return (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)

RECOMMENDATIONS = {
    "LOW_MILK_VS_EXPECTED": [
        "Check milking equipment and milking routine consistency.",
        "Review ration consistency and water access.",
        "Compare to herd peers at the same DIM (lactation stage).",
    ],
    "LOW_7DAY_AVG_MILK": [
        "Investigate sustained decline: nutrition, health, lameness.",
        "If available, check SCC/mastitis indicators and udder health.",
    ],
    "SUDDEN_DROP_MILK_3D": [
        "Treat as urgent: check illness, mastitis, stress, or feed disruption.",
    ],
    "VERY_LOW_FEED": [
        "Inspect feed availability, bunk space, and waterers.",
        "Check for illness or pain reducing appetite.",
    ],
    "LOW_FEED": [
        "Review ration, feeding schedule, sorting, and water access.",
        "Consider heat stress mitigation if THI elevated.",
    ],
    "FEED_DROP_3D": [
        "Early warning: appetite drop often precedes milk drop.",
        "Check for fever, digestive issues, lameness, or feed changes.",
    ],
    "HIGH_TEMP_FEVER": [
        "Possible fever: isolate and perform vet assessment.",
        "Check for infection, mastitis, respiratory illness.",
    ],
    "ELEVATED_TEMP": [
        "Monitor closely; if it persists, assess health status.",
    ],
    "LOW_RUMINATION": [
        "Possible rumen upset/illness: assess diet fiber, health, and lameness.",
    ],
    "REDUCED_RUMINATION": [
        "Monitor rumination + appetite; review feed quality and consistency.",
    ],
    "RUMINATION_DROP_3D": [
        "Early warning: rumination drop can precede clinical signs.",
    ],
    "HEAT_STRESS_THI_HIGH": [
        "Activate heat stress protocol: shade/fans/misters.",
        "Feed during cooler hours and ensure extra water access.",
    ],
    "HEAT_STRESS_THI_MODERATE": [
        "Monitor for heat stress; ensure shade and water access.",
    ],
    "LIKELY_APPETITE_ISSUE": [
        "Pattern suggests appetite issue: prioritize feed/water + health check.",
    ],
    "LIKELY_HEALTH_ISSUE": [
        "Pattern suggests health issue: temperature + production abnormal.",
    ],
}

def risk_score_from_signals(
    *,
    expected_liters: float,
    pct_diff: float,  # fraction (e.g. -0.25)
    temp: Optional[float],
    feed: Optional[float],
    rum: Optional[float],
    thi: Optional[float],
    drop3_milk: Optional[float],
    feed_drop3: Optional[float],
    rum_drop3: Optional[float],
) -> float:
    score = 0.0

    if expected_liters > 0:
        score += min(50.0, max(0.0, (-pct_diff) * 100.0) * 1.6)

    if temp is not None:
        if temp >= 39.5:
            score += 25.0
        elif temp >= 39.2:
            score += 12.0

    if feed is not None:
        if feed < 14.0:
            score += 18.0
        elif feed < 16.5:
            score += 10.0

    if rum is not None:
        if rum < 300:
            score += 18.0
        elif rum < 360:
            score += 10.0

    if thi is not None:
        if thi >= 80:
            score += 12.0
        elif thi >= 72:
            score += 6.0

    if drop3_milk is not None and drop3_milk < -0.15:
        score += 10.0
    if feed_drop3 is not None and feed_drop3 < -0.12:
        score += 6.0
    if rum_drop3 is not None and rum_drop3 < -0.12:
        score += 6.0

    return float(max(0.0, min(100.0, round(score, 1))))

def score_cow(
    *,
    cow,
    latest,
    expected_liters: float,
    recent: List,
) -> Dict[str, Any]:
    actual = float(latest.milk_liters)
    diff = actual - expected_liters
    pct = (diff / expected_liters) if expected_liters > 0 else 0.0

    feed = latest.feed_intake_kg
    temp = latest.body_temp_c
    rum = latest.rumination_min
    act = latest.activity_index
    eat = latest.eating_min
    amb = latest.ambient_temp_c
    hum = latest.humidity_pct
    thi = compute_thi(amb, hum)

    avg7 = None
    drop3_pct = None
    feed_drop3_pct = None
    rum_drop3_pct = None

    if len(recent) >= 7:
        last7 = recent[-7:]
        avg7 = sum(float(r.milk_liters) for r in last7) / 7.0

    if len(recent) >= 3:
        first = float(recent[-3].milk_liters)
        last = float(recent[-1].milk_liters)
        if first > 0:
            drop3_pct = (last - first) / first

        f0 = recent[-3].feed_intake_kg
        f1 = recent[-1].feed_intake_kg
        if f0 is not None and f1 is not None and f0 > 0:
            feed_drop3_pct = (float(f1) - float(f0)) / float(f0)

        r0 = recent[-3].rumination_min
        r1 = recent[-1].rumination_min
        if r0 is not None and r1 is not None and r0 > 0:
            rum_drop3_pct = (float(r1) - float(r0)) / float(r0)

    reasons: List[str] = []
    status = "OK"

    if expected_liters <= 0:
        status = "UNKNOWN"
        reasons.append("NO_BASELINE")
    else:
        if pct < -0.20:
            status = "FLAG"
            reasons.append("LOW_MILK_VS_EXPECTED")
        elif pct < -0.10:
            status = "WARNING"
            reasons.append("SLIGHTLY_LOW_MILK")

    if avg7 is not None and expected_liters > 0 and avg7 < expected_liters * 0.85:
        status = "FLAG"
        reasons.append("LOW_7DAY_AVG_MILK")

    if drop3_pct is not None and drop3_pct < -0.15:
        status = "FLAG"
        reasons.append("SUDDEN_DROP_MILK_3D")

    if feed is not None:
        if feed < 14.0:
            status = "FLAG"
            reasons.append("VERY_LOW_FEED")
        elif feed < 16.5 and status != "FLAG":
            status = "WARNING" if status == "OK" else status
            reasons.append("LOW_FEED")

    if feed_drop3_pct is not None and feed_drop3_pct < -0.12:
        status = "WARNING" if status != "FLAG" else status
        reasons.append("FEED_DROP_3D")

    if temp is not None:
        if temp >= 39.5:
            status = "FLAG"
            reasons.append("HIGH_TEMP_FEVER")
        elif temp >= 39.2 and status != "FLAG":
            status = "WARNING" if status == "OK" else status
            reasons.append("ELEVATED_TEMP")

    if rum is not None:
        if rum < 300:
            status = "FLAG"
            reasons.append("LOW_RUMINATION")
        elif rum < 360 and status != "FLAG":
            status = "WARNING" if status == "OK" else status
            reasons.append("REDUCED_RUMINATION")

    if rum_drop3_pct is not None and rum_drop3_pct < -0.12:
        status = "WARNING" if status != "FLAG" else status
        reasons.append("RUMINATION_DROP_3D")

    if thi is not None:
        if thi >= 80:
            status = "WARNING" if status != "FLAG" else status
            reasons.append("HEAT_STRESS_THI_HIGH")
        elif thi >= 72:
            reasons.append("HEAT_STRESS_THI_MODERATE")

    if "LOW_MILK_VS_EXPECTED" in reasons and ("LOW_FEED" in reasons or "FEED_DROP_3D" in reasons):
        reasons.append("LIKELY_APPETITE_ISSUE")
    if "LOW_MILK_VS_EXPECTED" in reasons and ("HIGH_TEMP_FEVER" in reasons or "ELEVATED_TEMP" in reasons):
        reasons.append("LIKELY_HEALTH_ISSUE")

    rscore = risk_score_from_signals(
        expected_liters=expected_liters,
        pct_diff=pct,
        temp=temp,
        feed=feed,
        rum=rum,
        thi=thi,
        drop3_milk=drop3_pct,
        feed_drop3=feed_drop3_pct,
        rum_drop3=rum_drop3_pct,
    )

    recs: List[str] = []
    for code in reasons:
        recs.extend(RECOMMENDATIONS.get(code, []))
    seen = set()
    recs_unique = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            recs_unique.append(r)

    dim_asof = days_in_milk(cow.last_calving_date, latest.date)

    return {
        "cow_id": cow.cow_id,
        "breed": cow.breed,
        "days_in_milk": dim_asof,
        "record_date": latest.date.isoformat(),

        "expected_liters": round(expected_liters, 2),
        "actual_liters": round(actual, 2),
        "difference_liters": round(diff, 2),
        "difference_pct": round(pct * 100.0, 1),

        "feed_intake_kg": feed,
        "body_temp_c": temp,
        "rumination_min": rum,
        "eating_min": eat,
        "activity_index": act,
        "ambient_temp_c": amb,
        "humidity_pct": hum,
        "thi": round(thi, 1) if thi is not None else None,

        "avg7_milk_liters": round(avg7, 2) if avg7 is not None else None,
        "drop3_milk_pct": round(drop3_pct * 100.0, 1) if drop3_pct is not None else None,
        "drop3_feed_pct": round(feed_drop3_pct * 100.0, 1) if feed_drop3_pct is not None else None,
        "drop3_rumination_pct": round(rum_drop3_pct * 100.0, 1) if rum_drop3_pct is not None else None,

        "status": status,
        "risk_score": rscore,
        "reasons": reasons,
        "recommendations": recs_unique,
        "reason_text": ", ".join(reasons) if reasons else "WITHIN_EXPECTED_RANGE",
        "last_calving_date": cow.last_calving_date.isoformat(),
    }
