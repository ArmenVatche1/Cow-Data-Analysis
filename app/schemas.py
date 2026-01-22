from __future__ import annotations
from datetime import date
from pydantic import BaseModel, Field
from typing import Optional

class CowCreate(BaseModel):
    cow_id: str = Field(..., min_length=1)
    breed: str = Field(default="Holstein")
    birth_date: date
    last_calving_date: date
    parity: Optional[int] = Field(default=None, ge=0, le=20)
    weight_kg: Optional[float] = Field(default=None, ge=0)

class DailyRecordCreate(BaseModel):
    cow_id: str = Field(..., min_length=1)
    date: date

    milk_liters: float = Field(..., ge=0)

    feed_intake_kg: Optional[float] = Field(default=None, ge=0)
    body_temp_c: Optional[float] = Field(default=None, ge=30, le=45)

    rumination_min: Optional[float] = Field(default=None, ge=0)
    eating_min: Optional[float] = Field(default=None, ge=0)
    activity_index: Optional[float] = Field(default=None, ge=0)

    ambient_temp_c: Optional[float] = Field(default=None)
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100)
