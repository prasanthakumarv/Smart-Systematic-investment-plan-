"""
app.py
─────────────────────────────────────────────────────────────────────────────
Smart SIP — FastAPI Backend
─────────────────────────────────────────────────────────────────────────────
Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    SIPRequest, ForecastRequest, GoalRequest,
    BacktestRequest, PortfolioRequest,
    SIPResponse, LivePriceResponse, HealthResponse,
)
from sip_engine       import run_sip, portfolio_summary
from price_simulator  import forecast, tick_price, backtest_smart_vs_standard
from analytics        import (
    xirr, rolling_stats, drawdown_analysis,
    sip_break_even, goal_planner, rca_analysis,
)

# ─── app setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Smart SIP API",
    description = "Backend for the Smart Systematic Investment Plan gold & silver app",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── health ───────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    return HealthResponse()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse()


# ─── live prices ──────────────────────────────────────────────────────────────
@app.get("/prices/live", response_model=LivePriceResponse, tags=["Prices"])
def live_prices():
    """
    Returns simulated live tick prices for gold and silver.
    Call every 3–5 seconds from the frontend for a live feel.
    """
    return LivePriceResponse(gold=tick_price("gold"), silver=tick_price("silver"))


@app.get("/prices/forecast", tags=["Prices"])
def price_forecast(
    metal:  str = Query("gold",  description="gold or silver"),
    months: int = Query(36,      ge=1, le=120),
    paths:  int = Query(200,     ge=10, le=1000),
):
    """
    Monte Carlo price forecast using Geometric Brownian Motion.
    Returns percentile bands (p5, p25, p50, p75, p95).
    """
    if metal not in ("gold", "silver"):
        raise HTTPException(400, "metal must be 'gold' or 'silver'")
    return forecast(metal=metal, months=months, paths=paths)


# ─── SIP calculator ──────────────────────────────────────────────────────────
@app.post("/sip/calculate", response_model=SIPResponse, tags=["SIP"])
def calculate_sip(req: SIPRequest):
    """
    Run a full Smart SIP or Standard SIP simulation.
    Returns value history, invested history, monthly detail, and return metrics.
    """
    result = run_sip(
        base_monthly    = req.base_monthly,
        years           = req.years,
        dip_sensitivity = req.dip_sensitivity,
        sip_type        = req.sip_type,
        metal           = req.metal,
    )
    return result.to_dict()


@app.get("/sip/calculate", tags=["SIP"])
def calculate_sip_get(
    monthly:  float = Query(500,    ge=10),
    years:    int   = Query(3,      ge=1, le=40),
    dip:      float = Query(10.0,   ge=0, le=30),
    sip_type: str   = Query("smart"),
    metal:    str   = Query("gold"),
):
    """
    GET version of the SIP calculator — easy to call from the browser.
    """
    result = run_sip(
        base_monthly    = monthly,
        years           = years,
        dip_sensitivity = dip,
        sip_type        = sip_type,
        metal           = metal,
    )
    return result.to_dict()


# ─── backtest ────────────────────────────────────────────────────────────────
@app.post("/sip/backtest", tags=["SIP"])
def backtest(req: BacktestRequest):
    """
    Run Smart SIP vs Standard SIP across multiple price scenarios (Monte Carlo).
    Returns the distribution of the Smart SIP advantage.
    """
    return backtest_smart_vs_standard(
        metal           = req.metal,
        base_monthly    = req.base_monthly,
        years           = req.years,
        dip_sensitivity = req.dip_sensitivity,
        paths           = req.paths,
    )


# ─── analytics ───────────────────────────────────────────────────────────────
@app.post("/analytics/rolling", tags=["Analytics"])
def rolling(prices: list[float], window: int = Query(12, ge=3, le=36)):
    """
    Compute rolling return, volatility, and Sharpe ratio for a price series.
    """
    if len(prices) < window + 1:
        raise HTTPException(400, f"Need at least {window+1} price points")
    return rolling_stats(prices, window=window)


@app.post("/analytics/drawdown", tags=["Analytics"])
def drawdown(prices: list[float]):
    """Max-drawdown, duration, and recovery analysis."""
    if len(prices) < 2:
        raise HTTPException(400, "Need at least 2 price points")
    return drawdown_analysis(prices)


@app.post("/analytics/rca", tags=["Analytics"])
def rca(prices: list[float], monthly: float = Query(500, ge=10)):
    """
    Rupee Cost Averaging vs fixed-quantity buying comparison.
    Shows why ₹-based SIP outperforms gram-based buying.
    """
    return rca_analysis(prices, monthly)


@app.get("/analytics/goal", tags=["Analytics"])
def goal(
    target:  float = Query(...,  ge=1000, description="Target corpus ₹"),
    years:   int   = Query(10,   ge=1, le=40),
    returns: float = Query(0.11, ge=0.01, le=0.50),
    infl:    float = Query(0.06, ge=0.0,  le=0.20),
):
    """
    Goal-based planning: how much to invest monthly to reach a target corpus.
    """
    return goal_planner(target, years, returns, infl)


@app.get("/analytics/breakeven", tags=["Analytics"])
def breakeven(
    monthly:       float = Query(500,   ge=10),
    annual_return: float = Query(0.11,  ge=0.01, le=0.50),
    current_price: float = Query(9156,  ge=100),
    buy_price:     float = Query(9000,  ge=100),
):
    """
    How many months until SIP corpus crosses total invested amount.
    """
    return sip_break_even(monthly, annual_return, current_price, buy_price)


# ─── portfolio ───────────────────────────────────────────────────────────────
@app.post("/portfolio/summary", tags=["Portfolio"])
def portfolio(req: PortfolioRequest):
    """
    Aggregate portfolio analytics: total value, P&L, weight per metal.
    """
    positions = [p.model_dump() for p in req.positions]
    return portfolio_summary(positions)


# ─── XIRR ────────────────────────────────────────────────────────────────────
@app.post("/analytics/xirr", tags=["Analytics"])
def compute_xirr(cashflows: list[float], dates: list[str]):
    """
    Extended IRR for irregular cashflows.
    dates format: YYYY-MM-DD
    cashflows: negative = outflow (investment), positive = inflow (redemption).
    """
    from datetime import date as dt_date
    try:
        parsed_dates = [dt_date.fromisoformat(d) for d in dates]
    except ValueError as e:
        raise HTTPException(400, f"Invalid date format: {e}")
    if len(cashflows) != len(parsed_dates):
        raise HTTPException(400, "cashflows and dates must have the same length")
    rate = xirr(cashflows, parsed_dates)
    return {"xirr": rate, "xirr_pct": round(rate * 100, 2)}
