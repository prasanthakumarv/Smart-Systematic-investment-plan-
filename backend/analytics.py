"""
analytics.py
─────────────────────────────────────────────────────────────────────────────
Portfolio & SIP analytics layer.
Libraries: Pandas, NumPy, SciPy

Functions
  • xirr              — Extended IRR (actual annual return on irregular cashflows)
  • rolling_stats      — Rolling mean / std / Sharpe for a return series
  • drawdown_analysis  — Max drawdown, duration, recovery
  • sip_break_even     — Minimum months until SIP breaks even at current price
  • goal_planner       — How much to invest monthly to reach a target corpus
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skew, kurtosis
from datetime import datetime, date
from typing import Union


# ─── XIRR ─────────────────────────────────────────────────────────────────────
def _npv(rate: float, cashflows: list[float], dates: list[date]) -> float:
    t0 = dates[0]
    return sum(
        cf / (1 + rate) ** ((d - t0).days / 365.25)
        for cf, d in zip(cashflows, dates)
    )


def xirr(cashflows: list[float], dates: list[Union[date, datetime]]) -> float:
    """
    Calculate the Extended Internal Rate of Return (XIRR) for irregular cashflows.

    cashflows: negative = investment outflow, positive = redemption / final value
    dates    : corresponding dates

    Returns annualised IRR (0.12 = 12 % p.a.)
    """
    dates = [d.date() if isinstance(d, datetime) else d for d in dates]
    try:
        return round(brentq(_npv, -0.999, 100.0, args=(cashflows, dates)), 6)
    except ValueError:
        return float("nan")


# ─── rolling statistics ───────────────────────────────────────────────────────
def rolling_stats(
    price_series: list[float],
    window: int = 12,
    risk_free: float = 0.065,   # 6.5 % p.a. RBI repo rate proxy
) -> dict:
    """
    Compute rolling mean return, volatility, and Sharpe ratio.
    Returns dict with lists suitable for JSON serialisation.
    """
    s  = pd.Series(price_series, dtype=float)
    r  = s.pct_change().dropna()

    rf_monthly = (1 + risk_free) ** (1 / 12) - 1

    roll_ret   = r.rolling(window).mean() * 12           # annualised
    roll_vol   = r.rolling(window).std()  * np.sqrt(12)  # annualised
    roll_sharpe= (roll_ret - risk_free) / roll_vol

    return {
        "window":       window,
        "returns":      r.round(6).tolist(),
        "roll_return":  roll_ret.round(4).fillna(0).tolist(),
        "roll_vol":     roll_vol.round(4).fillna(0).tolist(),
        "roll_sharpe":  roll_sharpe.round(4).fillna(0).tolist(),
        "skewness":     round(float(skew(r)),     4),
        "kurtosis":     round(float(kurtosis(r)), 4),
        "ann_return":   round(float(r.mean() * 12),           4),
        "ann_vol":      round(float(r.std() * np.sqrt(12)),   4),
        "sharpe":       round(float((r.mean() * 12 - risk_free) /
                                    (r.std() * np.sqrt(12))), 4),
    }


# ─── drawdown analysis ────────────────────────────────────────────────────────
def drawdown_analysis(price_series: list[float]) -> dict:
    """
    Compute maximum drawdown, its duration, and recovery info.
    """
    s        = pd.Series(price_series, dtype=float)
    peak     = s.expanding().max()
    dd       = (s - peak) / peak          # drawdown fraction (≤ 0)
    max_dd   = float(dd.min())
    max_dd_i = int(dd.idxmin())

    # find the peak before the trough
    peak_i   = int(peak.iloc[:max_dd_i + 1].idxmax())

    # find recovery (price ≥ peak again after trough)
    recovered = s.iloc[max_dd_i:] >= s.iloc[peak_i]
    rec_i     = int(recovered.idxmax()) if recovered.any() else -1

    drawdown_series = dd.round(6).tolist()

    return {
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "peak_month":        peak_i,
        "trough_month":      max_dd_i,
        "recovery_month":    rec_i,
        "drawdown_duration": max_dd_i - peak_i,
        "recovery_months":   (rec_i - max_dd_i) if rec_i >= 0 else None,
        "drawdown_series":   drawdown_series,
    }


# ─── SIP break-even ──────────────────────────────────────────────────────────
def sip_break_even(
    monthly: float,
    annual_return: float,
    current_price: float,
    buy_price: float,
) -> dict:
    """
    Returns the minimum number of months until the SIP corpus
    exceeds total invested, given the current price trend.
    """
    months_list = range(1, 361)
    for n in months_list:
        mr = annual_return / 12
        fv = monthly * (((1 + mr) ** n - 1) / mr) * (1 + mr)
        invested = monthly * n
        if fv >= invested:
            return {"break_even_months": n, "invested": round(invested, 2),
                    "expected_value": round(fv, 2)}
    return {"break_even_months": None, "invested": None, "expected_value": None}


# ─── goal planner ────────────────────────────────────────────────────────────
def goal_planner(
    target_corpus: float,
    years:         int,
    annual_return: float = 0.11,
    inflation:     float = 0.06,
) -> dict:
    """
    Calculate monthly SIP amount needed to reach `target_corpus` in `years`.
    Adjusts for inflation to show real value.
    """
    n   = years * 12
    mr  = annual_return / 12
    # PMT formula: FV = PMT × [((1+r)^n - 1) / r] × (1+r)
    denom  = ((1 + mr) ** n - 1) / mr * (1 + mr)
    pmt    = target_corpus / denom

    real_corpus = target_corpus / (1 + inflation) ** years
    total_inv   = pmt * n

    return {
        "target_corpus":       round(target_corpus, 2),
        "years":               years,
        "monthly_sip_needed":  round(pmt, 2),
        "total_invested":      round(total_inv, 2),
        "expected_gain":       round(target_corpus - total_inv, 2),
        "inflation_adj_value": round(real_corpus, 2),
        "assumed_return_pct":  round(annual_return * 100, 1),
    }


# ─── rupee cost averaging analysis ───────────────────────────────────────────
def rca_analysis(
    prices:  list[float],
    monthly: float,
) -> dict:
    """
    Rupee Cost Averaging (RCA) — show why investing a fixed ₹ amount
    is better than buying a fixed quantity each month.
    """
    prices_arr = np.array(prices, dtype=float)
    n          = len(prices_arr)

    # RCA: fixed ₹ → variable units
    rca_units  = monthly / prices_arr
    rca_total  = rca_units.cumsum()
    rca_avg    = (monthly * np.arange(1, n + 1)) / rca_total

    # Fixed quantity: buy same grams each month at varying cost
    fixed_grams   = monthly / prices_arr[0]          # quantity fixed at month-1 price
    fixed_units   = np.full(n, fixed_grams)
    fixed_invested= fixed_units * prices_arr
    fixed_total   = fixed_units.cumsum()
    fixed_avg     = (fixed_invested.cumsum()) / fixed_total

    return {
        "rca_avg_cost":   round(float(rca_avg[-1]),   2),
        "fixed_avg_cost": round(float(fixed_avg[-1]), 2),
        "rca_advantage":  round(float(fixed_avg[-1] - rca_avg[-1]), 2),
        "rca_avg_series": rca_avg.round(2).tolist(),
        "fixed_avg_series": fixed_avg.round(2).tolist(),
        "n_months": n,
    }
