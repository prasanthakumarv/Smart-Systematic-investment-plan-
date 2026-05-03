# 💰 Smart SIP — Systematic Investment Plan for Gold & Silver

> A mobile-first investment app with a **Python data-science backend** powering intelligent dip-buying, Monte Carlo forecasting, and real-time portfolio analytics.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                            │
│                                                                     │
│   smart_sip.html                                                    │
│   ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────┐   │
│   │   Home    │  │  SmartSIP  │  │Portfolio │  │  Transactions │   │
│   │ Live Gold │  │ Calculator │  │  Pie     │  │   History     │   │
│   │  Price    │  │  Chart.js  │  │  Chart   │  │               │   │
│   └─────┬─────┘  └─────┬──────┘  └────┬─────┘  └───────────────┘   │
└─────────┼──────────────┼──────────────┼───────────────────────────┘
          │  REST API    │              │
          ▼   (JSON)     ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend  (app.py)                        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │  /prices/*   │  │   /sip/*     │  │    /analytics/*         │   │
│  │  live tick   │  │  calculate   │  │  goal / breakeven       │   │
│  │  forecast    │  │  backtest    │  │  rolling / drawdown     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────────┘   │
└─────────┼────────────────┼─────────────────────┼───────────────────┘
          ▼                ▼                     ▼
┌──────────────┐  ┌─────────────────┐  ┌───────────────────────────┐
│price_simulat │  │  sip_engine.py  │  │      analytics.py         │
│    or.py     │  │                 │  │                           │
│  NumPy GBM   │  │ NumPy vectorised│  │ Pandas rolling stats      │
│  SciPy norm  │  │ Pandas detail   │  │ SciPy XIRR / brentq      │
│  Mean-rev    │  │ Smart dip mult  │  │ Drawdown / Sharpe / RCA   │
└──────────────┘  └─────────────────┘  └───────────────────────────┘
```

---

## 🚀 How It Works

### 1. Smart SIP Engine (`backend/sip_engine.py`)
Each month the engine detects price dips vs trend and scales investment up to 2.5× base — buying more units when gold/silver is cheap.

### 2. Price Simulator (`backend/price_simulator.py`)
Geometric Brownian Motion + mean-reversion + seasonal (Diwali) adjustment. Monte Carlo with 200 paths for honest forecast bands.

### 3. Analytics Layer (`backend/analytics.py`)
XIRR, Rolling Sharpe, Max Drawdown, Goal Planner, Rupee Cost Averaging — all powered by Pandas + SciPy.

### 4. REST API (`backend/app.py`)
FastAPI + Pydantic v2. Interactive docs at `/docs`.

| Endpoint | Description |
|----------|-------------|
| `GET /prices/live` | Simulated live price tick |
| `GET /prices/forecast` | Monte Carlo forecast (p5–p95) |
| `GET /sip/calculate` | Full SIP simulation |
| `POST /sip/backtest` | Smart vs Standard across many scenarios |
| `GET /analytics/goal` | Monthly SIP needed for target corpus |
| `POST /analytics/xirr` | Extended IRR on cashflows |
| `POST /analytics/rolling` | Rolling Sharpe / volatility |
| `POST /analytics/drawdown` | Max-drawdown analysis |
| `POST /portfolio/summary` | Portfolio P&L aggregation |

---

## 📁 Project Structure

```
Smart-Systematic-investment-plan-/
├── smart_sip.html              ← Frontend (mobile UI, Chart.js)
├── generate_data.py            ← Generate price_history.csv
│
├── backend/
│   ├── app.py                  ← FastAPI routes
│   ├── sip_engine.py           ← SIP calculation (NumPy + Pandas)
│   ├── price_simulator.py      ← GBM simulation (NumPy + SciPy)
│   ├── analytics.py            ← Portfolio analytics (Pandas + SciPy)
│   ├── models.py               ← Pydantic v2 schemas
│   └── requirements.txt
│
├── data/
│   └── price_history.csv       ← 60 months gold & silver prices
│
└── notebooks/
    └── sip_analysis.ipynb      ← Jupyter analysis & charts
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/prasanthakumarv/Smart-Systematic-investment-plan-.git
cd Smart-Systematic-investment-plan-
pip install -r backend/requirements.txt
python generate_data.py
uvicorn backend.app:app --reload --port 8000
# API docs → http://localhost:8000/docs
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JS, Chart.js |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Numerics | NumPy 1.26, SciPy 1.13 |
| Data | Pandas 2.2 |
| Schemas | Pydantic v2 |
| Notebooks | Jupyter, Matplotlib |

---

**Author:** Prasantha Kumar V · [GitHub](https://github.com/prasanthakumarv)
