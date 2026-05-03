"""
price_simulator.py
─────────────────────────────────────────────────────────────────────────────
Realistic gold & silver price simulation using:
  • Geometric Brownian Motion  (GBM)  — standard financial model
  • Mean-reversion overlay            — prevents price from drifting wild
  • Seasonal adjustment               — mild Q4 / festive bump for India

Libraries: NumPy, SciPy, Pandas
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Optional

# ─── calibrated market parameters ────────────────────────────────────────────
PARAMS: dict[str, dict] = {
    "gold": {
        "mu":        0.11,       # annual drift  (historical long-run ~11 %)
        "sigma":     0.14,       # annual vol    (14 % is reasonable for gold)
        "mean_rev":  0.08,       # mean-reversion speed (Ornstein-Uhlenbeck)
        "spot":      9_156,      # ₹/gram  live spot (updated via API)
        "seasonal":  {10: 0.008, 11: 0.010, 12: 0.006},  # Diwali / wedding Q
    },
    "silver": {
        "mu":        0.09,
        "sigma":     0.22,       # silver is more volatile
        "mean_rev":  0.10,
        "spot":      1_043,
        "seasonal":  {10: 0.005, 11: 0.007},
    },
}


# ─── GBM simulation ───────────────────────────────────────────────────────────
def simulate_gbm(
    metal:  str  = "gold",
    months: int  = 36,
    paths:  int  = 1,
    seed:   int  = 0,
    spot:   Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulate `paths` price paths for `months` using Geometric Brownian Motion
    with a mean-reversion overlay.

    Returns a DataFrame of shape (months, paths) with DatetimeIndex.
    """
    p     = PARAMS[metal]
    S0    = spot or p["spot"]
    mu    = p["mu"]
    sigma = p["sigma"]
    kappa = p["mean_rev"]         # mean-reversion coefficient
    dt    = 1 / 12                # monthly steps

    rng   = np.random.default_rng(seed)
    Z     = rng.standard_normal((months, paths))

    prices = np.zeros((months + 1, paths))
    prices[0] = S0

    long_run = S0 * np.exp(mu * np.arange(months + 1) * dt)  # trend line

    for t in range(1, months + 1):
        drift  = (mu - 0.5 * sigma ** 2) * dt
        diffus = sigma * np.sqrt(dt) * Z[t - 1]
        mr     = kappa * (np.log(long_run[t]) - np.log(prices[t - 1])) * dt
        prices[t] = prices[t - 1] * np.exp(drift + diffus + mr)

        # seasonal adjustment
        month_no = (t % 12) or 12
        prices[t] *= 1 + p["seasonal"].get(month_no, 0.0)

    start = datetime.today().replace(day=1)
    idx   = pd.date_range(start, periods=months, freq="MS")
    df    = pd.DataFrame(
        prices[1:],
        index   = idx,
        columns = [f"path_{i+1}" for i in range(paths)],
    )
    df.index.name = "date"
    return df.round(2)


def forecast(
    metal:  str = "gold",
    months: int = 36,
    paths:  int = 200,
    seed:   int = 42,
) -> dict:
    """
    Run a Monte Carlo forecast and return percentile bands + stats.
    """
    df = simulate_gbm(metal=metal, months=months, paths=paths, seed=seed)

    p5   = df.quantile(0.05, axis=1).round(2)
    p25  = df.quantile(0.25, axis=1).round(2)
    p50  = df.quantile(0.50, axis=1).round(2)
    p75  = df.quantile(0.75, axis=1).round(2)
    p95  = df.quantile(0.95, axis=1).round(2)

    labels = [d.strftime("%b %Y") for d in df.index]

    # probability that final price > current spot
    final = df.iloc[-1]
    spot  = PARAMS[metal]["spot"]
    prob_up = round((final > spot).mean() * 100, 1)

    return {
        "metal":    metal,
        "months":   months,
        "paths":    paths,
        "spot":     spot,
        "labels":   labels,
        "p5":       p5.tolist(),
        "p25":      p25.tolist(),
        "p50":      p50.tolist(),
        "p75":      p75.tolist(),
        "p95":      p95.tolist(),
        "prob_positive_return": prob_up,
        "expected_final":       round(float(p50.iloc[-1]), 2),
    }


# ─── live tick simulation ─────────────────────────────────────────────────────
def tick_price(metal: str = "gold") -> dict:
    """
    Produce a realistic intraday tick (±0.5 % of last price).
    Called on every /prices/live poll.
    """
    p      = PARAMS[metal]
    now    = datetime.utcnow()
    seed   = int(now.timestamp() / 3)          # changes every 3 s
    rng    = np.random.default_rng(seed)

    pct    = rng.normal(0, 0.002)              # 0.2 % intraday vol
    new_px = round(p["spot"] * (1 + pct), 2)
    p["spot"] = new_px                         # stateful update

    change_pct = round(pct * 100, 3)
    return {
        "metal":      metal,
        "price":      new_px,
        "change_pct": change_pct,
        "timestamp":  now.isoformat(),
    }


# ─── historical back-test helper ─────────────────────────────────────────────
def backtest_smart_vs_standard(
    metal:           str   = "gold",
    base_monthly:    float = 500,
    years:           int   = 5,
    dip_sensitivity: float = 10,
    paths:           int   = 50,
    seed:            int   = 99,
) -> dict:
    """
    Run multiple price paths and compare Smart SIP vs Standard across all of them.
    Returns distribution of advantage (₹ gained).
    """
    from sip_engine import _compute_dip_multiplier

    p      = PARAMS[metal]
    n      = years * 12
    results = {"advantages": [], "smart_rets": [], "std_rets": []}

    df = simulate_gbm(metal=metal, months=n, paths=paths, seed=seed,
                      spot=p["spot"])

    for col in df.columns:
        prices = df[col].values

        std_units = np.cumsum(np.full(n, base_monthly) / prices)
        std_final = float(std_units[-1] * prices[-1])
        std_inv   = base_monthly * n

        mult      = _compute_dip_multiplier(prices, dip_sensitivity)
        inv_each  = base_monthly * mult
        sm_units  = np.cumsum(inv_each / prices)
        sm_final  = float(sm_units[-1] * prices[-1])
        sm_inv    = float(inv_each.sum())

        results["advantages"].append(round(sm_final - std_final, 2))
        results["smart_rets"].append(round((sm_final / sm_inv - 1) * 100, 2))
        results["std_rets"].append  (round((std_final / std_inv - 1) * 100, 2))

    adv = np.array(results["advantages"])
    return {
        "paths":           paths,
        "metal":           metal,
        "years":           years,
        "mean_advantage":  round(float(adv.mean()), 2),
        "p25_advantage":   round(float(np.percentile(adv, 25)), 2),
        "p75_advantage":   round(float(np.percentile(adv, 75)), 2),
        "pct_paths_smart_wins": round((adv > 0).mean() * 100, 1),
        "avg_smart_return_pct": round(float(np.mean(results["smart_rets"])), 2),
        "avg_std_return_pct":   round(float(np.mean(results["std_rets"])),   2),
        "advantages":           results["advantages"],
    }
