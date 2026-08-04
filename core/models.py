"""
Core data models for the StatArb Engine.
Dataclasses keep this schema-first and serialisable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


class SignalState(str, Enum):
    FLAT    = "FLAT"
    ENTER   = "ENTER"
    HOLD    = "HOLD"
    EXIT    = "EXIT"


@dataclass
class PairConfig:
    """Static configuration for a cointegrated pair."""
    ticker_x: str
    ticker_y: str
    # OLS: y = hedge_ratio * x + intercept + residual
    hedge_ratio: float
    intercept: float
    half_life_days: float          # Ornstein–Uhlenbeck mean-reversion speed
    entry_z: float    = 2.0        # z-score threshold to enter
    exit_z: float     = 0.25       # z-score threshold to exit
    stop_z: float     = 3.5        # z-score stop-loss
    lookback: int     = 60         # rolling window for z-score normalisation


@dataclass
class SpreadSnapshot:
    """One timestep of spread state."""
    timestamp: datetime
    spread: float
    z_score: float
    spread_mean: float
    spread_std: float
    signal: SignalState = SignalState.FLAT


@dataclass
class Trade:
    """A round-trip trade on a pair."""
    pair_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    side: Side                    # LONG spread = long X, short Y
    entry_z: float
    exit_z: Optional[float]
    entry_price_x: float
    entry_price_y: float
    exit_price_x: Optional[float]
    exit_price_y: Optional[float]
    qty_x: float
    qty_y: float
    gross_pnl: float  = 0.0
    commission:  float = 0.0
    slippage:    float = 0.0
    net_pnl: float    = 0.0
    is_open: bool     = True


@dataclass
class PortfolioSnapshot:
    """Aggregate portfolio state at one point in time."""
    timestamp: datetime
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
    num_open_trades: int
    daily_pnl: float
    cumulative_pnl: float
    drawdown: float
    sharpe_rolling: float = 0.0
