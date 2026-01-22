from __future__ import annotations

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from .db import Base

class Cow(Base):
    __tablename__ = "cows"

    id = Column(Integer, primary_key=True, index=True)
    cow_id = Column(String, unique=True, index=True, nullable=False)

    breed = Column(String, nullable=False, default="Holstein")
    birth_date = Column(Date, nullable=False)
    last_calving_date = Column(Date, nullable=False)

    parity = Column(Integer, nullable=True)  # number of calvings
    weight_kg = Column(Float, nullable=True)

    records = relationship("DailyRecord", back_populates="cow", cascade="all, delete-orphan")
    scans = relationship("ScanEvent", back_populates="cow", cascade="all, delete-orphan")

class DailyRecord(Base):
    __tablename__ = "daily_records"

    id = Column(Integer, primary_key=True, index=True)
    cow_id = Column(String, ForeignKey("cows.cow_id"), nullable=False)
    date = Column(Date, nullable=False)

    milk_liters = Column(Float, nullable=False)

    feed_intake_kg = Column(Float, nullable=True)
    body_temp_c = Column(Float, nullable=True)

    rumination_min = Column(Float, nullable=True)
    eating_min = Column(Float, nullable=True)
    activity_index = Column(Float, nullable=True)

    ambient_temp_c = Column(Float, nullable=True)
    humidity_pct = Column(Float, nullable=True)

    cow = relationship("Cow", back_populates="records")

    __table_args__ = (
        UniqueConstraint("cow_id", "date", name="uix_cow_date"),
        Index("idx_rec_cow_date", "cow_id", "date"),
    )

class ExpectedMilkBaseline(Base):
    __tablename__ = "expected_milk_baseline"

    id = Column(Integer, primary_key=True, index=True)
    breed = Column(String, nullable=False, default="ALL")

    dim_min = Column(Integer, nullable=False)
    dim_max = Column(Integer, nullable=False)

    expected_liters = Column(Float, nullable=False)
    std_dev = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_baseline_breed_dim", "breed", "dim_min", "dim_max"),
    )

class ScanEvent(Base):
    __tablename__ = "scan_events"

    id = Column(Integer, primary_key=True, index=True)
    cow_id = Column(String, ForeignKey("cows.cow_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    status = Column(String, nullable=False)
    reasons = Column(String, nullable=True)  # comma-separated

    risk_score = Column(Float, nullable=True)

    expected_liters = Column(Float, nullable=True)
    actual_liters = Column(Float, nullable=True)
    difference_pct = Column(Float, nullable=True)

    cow = relationship("Cow", back_populates="scans")

    __table_args__ = (
        Index("idx_scan_cow_time", "cow_id", "timestamp"),
    )
