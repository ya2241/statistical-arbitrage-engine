"""
Market data layer.
Tries yfinance first; falls back to a realistic synthetic generator
so the project always runs deterministically in a CI / offline environment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ── Synthetic data (always available) ─────────────────────────────────────────

def _ou_process(n: int, theta_daily: float, mu: float, sigma: float,
                x0: float) -> np.ndarray:
    """
    Exact simulation of Ornstein-Uhlenbeck in discrete daily steps.
    theta_daily: mean-reversion rate per trading day (= ln(2) / half_life_days)
    Stationary variance: σ² / (2θ)
    """
    x = np.empty(n)
    x[0] = x0
    e_theta = np.exp(-theta_daily)
    noise_std = sigma * np.sqrt((1 - np.exp(-2 * theta_daily)) / (2 * theta_daily))
    for i in range(1, n):
        x[i] = x[i-1] * e_theta + mu * (1 - e_theta) + noise_std * np.random.randn()
    return x


def generate_cointegrated_pair(
    ticker_x: str,
    ticker_y: str,
    n_days: int = 756,          # 3 years
    start_price_x: float = 100.0,
    hedge_ratio: float = 1.5,
    half_life_days: float = 15.0,
    spread_vol: float = 2.0,
    daily_vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate two cointegrated price series.
    Y = hedge_ratio * X + spread,  where spread is OU mean-reverting.
    theta_daily = ln(2) / half_life_days so that E[τ_reversion] = half_life_days.
    """
    np.random.seed(seed)

    # Geometric Brownian Motion for X
    log_returns_x = np.random.normal(0.0005, daily_vol, n_days)
    price_x = start_price_x * np.exp(np.cumsum(log_returns_x))

    # OU spread: theta in daily units
    theta_daily = np.log(2) / half_life_days
    spread = _ou_process(n_days, theta_daily=theta_daily, mu=0,
                         sigma=spread_vol, x0=0)

    # Y is cointegrated with X
    price_y = hedge_ratio * price_x + spread
    price_y = np.maximum(price_y, 1.0)  # keep positive

    dates = pd.bdate_range(end=datetime.today(), periods=n_days)

    df = pd.DataFrame({
        "timestamp": np.concatenate([dates, dates]),
        "ticker":    [ticker_x] * n_days + [ticker_y] * n_days,
        "close":     np.concatenate([price_x, price_y]),
        "open":      np.concatenate([price_x * 0.999, price_y * 0.999]),
        "high":      np.concatenate([price_x * 1.005, price_y * 1.005]),
        "low":       np.concatenate([price_x * 0.995, price_y * 0.995]),
        "volume":    np.random.randint(1_000_000, 10_000_000, n_days * 2),
    })
    return df


def build_universe(seed: int = 99) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """
    Build a synthetic 5-pair universe.
    Returns: (pair_list, prices_df)
    """
    pairs_cfg = [
        # (x, y, hr,  hl,   vol,  seed)
        ("AAA", "BBB", 1.2,  12, 1.5, 10),
        ("CCC", "DDD", 0.8,  20, 2.0, 20),
        ("EEE", "FFF", 1.5,  10, 1.8, 30),
        ("GGG", "HHH", 1.1,  25, 2.5, 40),
        ("III", "JJJ", 0.9,  18, 1.2, 50),
    ]
    all_dfs = []
    pair_list = []
    for x, y, hr, hl, vol, s in pairs_cfg:
        df = generate_cointegrated_pair(x, y, hedge_ratio=hr,
                                        half_life_days=hl,
                                        spread_vol=vol, seed=s)
        all_dfs.append(df)
        pair_list.append((x, y))

    prices = pd.concat(all_dfs, ignore_index=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    return pair_list, prices


# ── Live data (yfinance) ───────────────────────────────────────────────────────

LIVE_PAIRS = [
    ("XOM",  "CVX"),   # Oil majors
    ("KO",   "PEP"),   # Beverage duopoly
    ("GS",   "MS"),    # Bulge-bracket banks
    ("MSFT", "GOOGL"), # Mega-cap tech
    ("WMT",  "TGT"),   # Retail
]

def fetch_live(tickers: list[str], period: str = "3y") -> Optional[pd.DataFrame]:
    """Download from yfinance; return None on failure."""
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            return None
        close = raw["Close"].copy()
        rows = []
        for t in tickers:
            if t not in close.columns:
                continue
            s = close[t].dropna().reset_index()
            s.columns = ["timestamp", "close"]
            s["ticker"] = t
            rows.append(s)
        df = pd.concat(rows, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["open"] = df["close"] * 0.999
        df["high"] = df["close"] * 1.005
        df["low"]  = df["close"] * 0.995
        df["volume"] = 5_000_000
        return df
    except Exception:
        return None
