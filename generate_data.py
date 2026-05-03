#!/usr/bin/env python3
"""generate_data.py — creates data/price_history.csv"""
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)
months = 60
dates  = pd.date_range("2020-01-01", periods=months, freq="MS")

def gbm(s0, mu, sigma, n, seed):
    rng = np.random.default_rng(seed)
    dt  = 1/12
    r   = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.standard_normal(n)
    return s0 * np.exp(np.cumsum(r))

gold   = gbm(4500, 0.11, 0.14, months, 42).round(2)
silver = gbm(600,  0.09, 0.22, months, 7).round(2)

df = pd.DataFrame({"date": dates, "gold_price": gold, "silver_price": silver})
df.to_csv("data/price_history.csv", index=False)
print(f"Saved {len(df)} rows to data/price_history.csv")
