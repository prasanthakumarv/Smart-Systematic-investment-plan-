"""
models.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 request and response schemas for the Smart SIP API.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List


# ─── requests ─────────────────────────────────────────────────────────────────
class SIPRequest(BaseModel):
    base_monthly:    float = Field(500,  ge=10,    le=1_000_000, description="Monthly investment ₹")
    years:           int   = Field(3,    ge=1,     le=40,        description="Investment horizon in years")
    dip_sensitivity: float = Field(10.0, ge=0,    le=30.0,      description="Smart SIP dip multiplier sensitivity")
    sip_type:        Literal["smart", "standard"] = "smart"
    metal:           Literal["gold", "silver"]    = "gold"


class ForecastRequest(BaseModel):
    metal:  Literal["gold", "silver"] = "gold"
    months: int  = Field(36, ge=1, le=120)
    paths:  int  = Field(200, ge=10, le=1000)
    seed:   int  = Field(42)


class GoalRequest(BaseModel):
    target_corpus: float = Field(..., ge=1000, description="Target amount ₹")
    years:         int   = Field(..., ge=1, le=40)
    annual_return: float = Field(0.11, ge=0.01, le=0.50)
    inflation:     float = Field(0.06, ge=0.0,  le=0.20)


class BacktestRequest(BaseModel):
    metal:           Literal["gold", "silver"] = "gold"
    base_monthly:    float = Field(500, ge=10, le=1_000_000)
    years:           int   = Field(5, ge=1, le=20)
    dip_sensitivity: float = Field(10.0, ge=0, le=30.0)
    paths:           int   = Field(50, ge=10, le=500)


class PortfolioPosition(BaseModel):
    metal:          str
    units:          float = Field(..., gt=0)
    avg_buy_price:  float = Field(..., gt=0)
    current_price:  float = Field(..., gt=0)


class PortfolioRequest(BaseModel):
    positions: List[PortfolioPosition]


# ─── responses ────────────────────────────────────────────────────────────────
class SIPResponse(BaseModel):
    metal:              str
    sip_type:           str
    base_monthly:       float
    years:              int
    dip_sensitivity:    float
    total_invested:     float
    final_value:        float
    total_units:        float
    return_pct:         float
    advantage_vs_std:   float
    labels:             List[str]
    value_history:      List[float]
    invested_history:   List[float]
    std_value_history:  List[float]
    monthly_detail:     List[dict]


class LivePriceResponse(BaseModel):
    gold:   dict
    silver: dict


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    message: str = "Smart SIP API is running"
