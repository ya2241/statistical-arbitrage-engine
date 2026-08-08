# ⟁ StatArb Engine
### Statistical Arbitrage · Pairs Trading · Python + DuckDB

A production-quality statistical arbitrage system.

---

## What this project demonstrates

| Skill Area | Implementation |
|---|---|
| **Statistics** | Engle-Granger + Johansen cointegration, ADF stationarity, OU half-life MLE |
| **Signal Generation** | Z-score mean-reversion with Kalman Filter dynamic hedge ratio |
| **Execution Simulation** | Commission, bid-ask spread, √ADV market impact (Almgren-Chriss), short borrow cost |
| **Risk Management** | Position limits, drawdown circuit breaker, parametric VaR, Kelly criterion, HHI concentration |
| **Backtesting** | Event-driven, no look-ahead, per-trade P&L attribution |
| **Data Engineering** | DuckDB schema, repository pattern, 10 window-function analytics queries |
| **Software Engineering** | Modular architecture, type hints, dataclasses, 27-test pytest suite |
| **Visualisation** | Interactive Plotly HTML dashboard (NAV, drawdown, Sharpe, monthly P&L, pair attribution) |

---

## Architecture

```
statarb/
├── core/
│   ├── models.py        # Typed dataclasses: PairConfig, Trade, SpreadSnapshot, PortfolioSnapshot
│   └── database.py      # DuckDB schema + repository + 10 analytics SQL queries
├── data/
│   └── fetcher.py       # Synthetic cointegrated pair generator (OU process) + yfinance live data
├── strategy/
│   ├── cointegration.py # Engle-Granger, Johansen, OLS hedge ratio, OU half-life MLE, pair scoring
│   └── signals.py       # Kalman Filter hedge ratio, rolling z-score, state machine signal generator
├── execution/
│   └── simulator.py     # Full cost model: commission + bid-ask + √ADV impact + borrow
├── risk/
│   └── manager.py       # Position limits, drawdown breaker, parametric VaR, Kelly, Herfindahl
├── backtest/
│   └── engine.py        # Event-driven orchestrator — wires all modules together
├── dashboard/
│   └── report.py        # Plotly interactive HTML report
├── sql/
│   └── analytics.sql    # Standalone SQL showcase: 10 production-quality DuckDB queries
├── tests/
│   └── test_core.py     # 27 pytest unit tests across all modules
└── main.py              # CLI entry point
```

---

## Quickstart

```bash
# Install dependencies
pip install numpy pandas scipy statsmodels scikit-learn yfinance duckdb plotly rich

# Run backtest with synthetic data (always works, no internet needed)
python main.py

# Run with live market data (yfinance)
python main.py --live

# Set custom starting capital
python main.py --capital 500000

# Run tests
python -m pytest tests/ -v
```

Output:
- Console: cointegration table + performance metrics (via `rich`)
- File: `statarb_report.html` — interactive dashboard, open in any browser

---

## The Statistical Pipeline

### 1. Cointegration Screening
For each candidate pair (X, Y):

**Engle-Granger two-step:**
```
Y_t = β·X_t + α + ε_t        [OLS]
ADF(ε_t) → H₀: unit root     [if rejected → cointegrated]
```

**Johansen trace test** (robustness check):
```
H₀: rank(Π) = 0   [no cointegration]
H₁: rank(Π) ≥ 1   [at least one cointegrating vector]
```

**OU half-life** (via exact MLE):
```
s_t = β·s_{t-1} + ε    →    θ = -ln(β)/dt    →    t₁/₂ = ln(2)/θ
```
Sweet spot: 5–60 days. Too fast → noise. Too slow → capital locked up.

### 2. Signal Generation

**Kalman Filter dynamic hedge ratio:**
```
State:  θ_t = [β_t, α_t]ᵀ    (hedge ratio + intercept)
Obs:    Y_t = [X_t, 1]·θ_t + ν_t
Update: θ_t = θ_{t-1} + K_t·(Y_t - Ŷ_t)
```
Adapts to regime changes in the cointegration relationship.

**Z-score normalisation:**
```
z_t = (s_t - μ̂_t) / σ̂_t      [rolling 60-day window]
```

**State machine:**
```
FLAT → |z| ≥ 2.0 → ENTER → HOLD → |z| ≤ 0.25 → EXIT → FLAT
                                 → |z| ≥ 3.5  → EXIT (stop-loss)
```

### 3. Execution Cost Model

| Cost Component | Model |
|---|---|
| Commission | $0.005/share, min $1.00/order |
| Bid-ask spread | 10 bps half-spread per fill |
| Market impact | η·σ·√(qty/ADV)  *(Almgren-Chriss)* |
| Short borrow | 50 bps/year, accrued daily |

### 4. Risk Controls
- Max 5 pairs open simultaneously
- Gross exposure cap: 5× capital
- Net exposure cap: 20% of capital
- Drawdown circuit breaker: halt at −15%
- 99% 1-day parametric VaR monitored

---

## SQL Showcase

`sql/analytics.sql` contains 10 production-quality DuckDB queries demonstrating:

- `WINDOW` functions (rolling Sharpe, drawdown, cumulative P&L waterfall)
- `WITH` CTEs for complex multi-step analytics
- Conditional aggregation (`CASE WHEN`) for win rate, streak analysis
- Time-series arithmetic (hold duration, monthly attribution)
- Self-join patterns for consecutive streak detection

All queries run against the live backtest database via `db.query(name)` or `db.raw(sql)`.


---

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
scipy >= 1.11
statsmodels >= 0.14
scikit-learn >= 1.3
yfinance >= 0.2
duckdb >= 0.9
plotly >= 5.18
rich >= 13.0
```
