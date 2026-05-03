"""
sip_engine.py
─────────────────────────────────────────────────────────────────────────────
Core Smart-SIP calculation engine.
Uses NumPy for vectorised maths and Pandas for time-series portfolio tracking.

Smart SIP logic
  • Standard SIP  : invest a fixed monthly amount regardless of price.
  • Smart  SIP    : detect price dips vs. a rolling trend and scale the
                   monthly investment up (max 2.5× base).  This buys
                   more units when the asset is cheap, improving average
                   cost over time.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

# ─── constants ────────────────────────────────────────────────────────────────
GOLD_ANNUAL_RETURN   = 0.11        # 11 % p.a. long-run assumption
SILVER_ANNUAL_RETURN = 0.09
PRICE_NOISE_SCALE    = 0.06        # amplitude of synthetic price noise
MAX_MULTIPLIER       = 2.5         # Smart SIP cap
BASE_GOLD_PRICE      = 7_000       # ₹ / gram starting price (seed)
BASE_SILVER_PRICE    = 950         # ₹ / gram starting price


# ─── helpers ──────────────────────────────────────────────────────────────────
def _generate_price_series(
    months: int,
    annual_return: float,
    base_price: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Produce a synthetic monthly price series with realistic drift + noise.
    Uses sinusoidal harmonics (no randomness so results are reproducible).
    """
    rng          = np.random.default_rng(seed)
    t            = np.arange(1, months + 1, dtype=float)
    trend        = base_price * np.power(1 + annual_return / 12, t)
    noise        = (
        np.sin(t * 0.70) * PRICE_NOISE_SCALE * trend
      + np.sin(t * 1.90) * (PRICE_NOISE_SCALE / 2) * trend
      + rng.normal(0, PRICE_NOISE_SCALE * 0.15, months) * trend
    )
    return np.maximum(trend + noise, base_price * 0.5)


def _compute_dip_multiplier(
    prices: np.ndarray,
    dip_sensitivity: float,
    cap: float = MAX_MULTIPLIER,
) -> np.ndarray:
    """
    For each month calculate the Smart SIP investment multiplier.
      multiplier = 1 + dip_fraction × dip_sensitivity   (capped at `cap`)
    dip_fraction = (trend - price) / trend   (0 when price ≥ trend)
    """
    months = len(prices)
    t      = np.arange(1, months + 1, dtype=float)
    # reconstruct pure trend without noise
    trend  = prices[0] * np.power(1 + GOLD_ANNUAL_RETURN / 12, t)
    dip    = np.maximum(0.0, (trend - prices) / trend)
    mult   = 1.0 + dip * dip_sensitivity
    return np.minimum(mult, cap)


# ─── dataclass results ────────────────────────────────────────────────────────
@dataclass
class SIPResult:
    metal:            str
    sip_type:         str
    base_monthly:     float
    years:            int
    dip_sensitivity:  float

    total_invested:   float = 0.0
    final_value:      float = 0.0
    total_units:      float = 0.0
    return_pct:       float = 0.0
    advantage_vs_std: float = 0.0      # only set for smart SIP

    # time-series snapshots (for charting)
    labels:           list[str]  = field(default_factory=list)
    value_history:    list[float] = field(default_factory=list)
    invested_history: list[float] = field(default_factory=list)
    std_value_history:list[float] = field(default_factory=list)   # comparison

    # monthly detail dataframe (serialised to list[dict] for JSON)
    monthly_detail:   list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─── main engine ──────────────────────────────────────────────────────────────
def run_sip(
    base_monthly:    float,
    years:           int,
    dip_sensitivity: float,
    sip_type:        Literal["smart", "standard"] = "smart",
    metal:           Literal["gold", "silver"]    = "gold",
) -> SIPResult:
    """
    Run a complete SIP simulation.

    Parameters
    ----------
    base_monthly    : monthly investment amount (₹)
    years           : investment horizon
    dip_sensitivity : 0–30  (how aggressively Smart SIP scales on dips)
    sip_type        : "smart" or "standard"
    metal           : "gold" or "silver"
    """
    n       = years * 12
    ann_ret = GOLD_ANNUAL_RETURN if metal == "gold" else SILVER_ANNUAL_RETURN
    b_price = BASE_GOLD_PRICE    if metal == "gold" else BASE_SILVER_PRICE

    prices = _generate_price_series(n, ann_ret, b_price)

    # ── standard SIP baseline ─────────────────────────────────────────────
    std_inv   = np.full(n, base_monthly)
    std_units = np.cumsum(std_inv / prices)
    std_vals  = std_units * prices

    if sip_type == "smart":
        mult      = _compute_dip_multiplier(prices, dip_sensitivity)
        inv_each  = base_monthly * mult
    else:
        inv_each  = np.full(n, base_monthly)

    units_each  = inv_each / prices
    cum_units   = np.cumsum(units_each)
    cum_invested= np.cumsum(inv_each)
    port_value  = cum_units * prices

    # ── snapshot indices for chart (≤12 points) ───────────────────────────
    snap_every = max(1, n // 12)
    snap_idx   = list(range(snap_every - 1, n, snap_every))
    if snap_idx[-1] != n - 1:
        snap_idx.append(n - 1)

    labels    = []
    val_hist  = []
    inv_hist  = []
    std_hist  = []
    for i in snap_idx:
        m = i + 1
        labels.append(f"{m}m" if m <= 12 else f"{m // 12}y")
        val_hist.append(round(float(port_value[i]),  2))
        inv_hist.append(round(float(cum_invested[i]),2))
        std_hist.append(round(float(std_vals[i]),    2))

    # ── monthly detail (last 24 months) ───────────────────────────────────
    detail_rows = []
    for i in range(max(0, n - 24), n):
        month_no = i + 1
        detail_rows.append({
            "month":        month_no,
            "price":        round(float(prices[i]),      2),
            "invested":     round(float(inv_each[i]),    2),
            "units_bought": round(float(units_each[i]),  6),
            "total_units":  round(float(cum_units[i]),   6),
            "portfolio_val":round(float(port_value[i]),  2),
            "multiplier":   round(float(inv_each[i] / base_monthly), 4),
        })

    final_val = float(port_value[-1])
    tot_inv   = float(cum_invested[-1])
    std_final = float(std_vals[-1])

    return SIPResult(
        metal            = metal,
        sip_type         = sip_type,
        base_monthly     = base_monthly,
        years            = years,
        dip_sensitivity  = dip_sensitivity,
        total_invested   = round(tot_inv,   2),
        final_value      = round(final_val, 2),
        total_units      = round(float(cum_units[-1]), 6),
        return_pct       = round((final_val / tot_inv - 1) * 100, 2),
        advantage_vs_std = round(final_val - std_final, 2),
        labels           = labels,
        value_history    = val_hist,
        invested_history = inv_hist,
        std_value_history= std_hist,
        monthly_detail   = detail_rows,
    )


# ─── portfolio aggregator ─────────────────────────────────────────────────────
def portfolio_summary(positions: list[dict]) -> dict:
    """
    Given a list of position dicts (metal, units, avg_buy_price, current_price),
    return portfolio analytics using Pandas.
    """
    df = pd.DataFrame(positions)
    df["cost"]       = df["units"] * df["avg_buy_price"]
    df["mkt_value"]  = df["units"] * df["current_price"]
    df["pnl"]        = df["mkt_value"] - df["cost"]
    df["pnl_pct"]    = (df["pnl"] / df["cost"] * 100).round(2)
    df["weight_pct"] = (df["mkt_value"] / df["mkt_value"].sum() * 100).round(2)

    return {
        "total_cost":      round(df["cost"].sum(),      2),
        "total_value":     round(df["mkt_value"].sum(), 2),
        "total_pnl":       round(df["pnl"].sum(),       2),
        "overall_pnl_pct": round(df["pnl"].sum() / df["cost"].sum() * 100, 2),
        "positions":       df.to_dict(orient="records"),
    }
