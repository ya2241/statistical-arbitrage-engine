"""
Execution simulator with realistic market microstructure costs.

Costs modelled:
  1. Commission: flat per-share + per-trade minimum
  2. Bid-ask spread: paid on entry and exit (half-spread per side)
  3. Market impact: square-root market impact model (Almgren-Chriss style)
  4. Borrow cost: for short leg (annualised, applied daily)

All costs are logged separately so attribution is clear.
"""
from __future__ import annotations
import numpy as np
import uuid
from datetime import datetime
from dataclasses import dataclass
from core.models import PairConfig, SignalState, Side, Trade


@dataclass
class ExecutionConfig:
    commission_per_share: float = 0.005      # $0.005/share (IB Tiered)
    min_commission:       float = 1.00       # minimum per order
    bid_ask_pct:          float = 0.001      # 10 bps round-trip half-spread
    market_impact_factor: float = 0.1        # Almgren-Chriss η coefficient
    borrow_rate_annual:   float = 0.005      # 50 bps/year short borrow cost
    capital_per_pair:     float = 100_000.0  # notional per pair trade


def _commission(qty: float, price: float, cfg: ExecutionConfig) -> float:
    gross = abs(qty) * cfg.commission_per_share
    return max(gross, cfg.min_commission)


def _slippage(qty: float, price: float, adv: float, cfg: ExecutionConfig) -> float:
    """
    Square-root market impact: impact ∝ η * σ * sqrt(qty / ADV)
    Bid-ask spread: paid on every fill.
    """
    participation = abs(qty) / max(adv, 1)
    impact = cfg.market_impact_factor * price * np.sqrt(participation)
    half_spread = price * cfg.bid_ask_pct / 2
    return (impact + half_spread) * abs(qty)


def _daily_borrow_cost(qty_short: float, price: float, cfg: ExecutionConfig) -> float:
    """Cost for one day of holding a short position."""
    return abs(qty_short) * price * cfg.borrow_rate_annual / 252


class ExecutionSimulator:
    """
    Walks through signal snapshots and manages the lifecycle of trades.

    Entry: allocates capital_per_pair equally across the two legs (dollar-neutral).
    Exit:  closes both legs, computes P&L net of all costs.
    """

    def __init__(self, cfg: ExecutionConfig | None = None):
        self.cfg = cfg or ExecutionConfig()
        self._open_trades: dict[str, tuple[Trade, str]] = {}  # pair_id → (trade, trade_id)

    def process(
        self,
        pair_cfg: PairConfig,
        snapshots,
        prices_x: dict[datetime, float],
        prices_y: dict[datetime, float],
        volumes_x: dict[datetime, float],
        volumes_y: dict[datetime, float],
    ) -> list[Trade]:
        """
        Process a sequence of SpreadSnapshots and return completed trades.
        """
        completed: list[Trade] = []
        half_capital = self.cfg.capital_per_pair / 2

        for snap in snapshots:
            ts = snap.timestamp
            px = prices_x.get(ts)
            py = prices_y.get(ts)
            vx = volumes_x.get(ts, 5_000_000)
            vy = volumes_y.get(ts, 5_000_000)

            if px is None or py is None:
                continue

            pair_id = f"{pair_cfg.ticker_x}_{pair_cfg.ticker_y}"

            # ── ENTRY ──────────────────────────────────────────────────────────
            if snap.signal == SignalState.ENTER and pair_id not in self._open_trades:
                # spread = Y - hr*X
                # z > 0: spread ABOVE mean → expect it to FALL → short spread
                #         short spread = short Y, long X  → Side.SHORT
                # z < 0: spread BELOW mean → expect it to RISE → long spread
                #         long spread = long Y, short X   → Side.LONG
                side = Side.SHORT if snap.z_score > 0 else Side.LONG

                qty_x = half_capital / px
                qty_y = half_capital / py

                # Dollar-neutral adjustment for hedge ratio
                # We want: qty_y * py ≈ pair_cfg.hedge_ratio * qty_x * px
                qty_y_adjusted = (pair_cfg.hedge_ratio * qty_x * px) / py

                # Execution costs — entry
                comm_x = _commission(qty_x, px, self.cfg)
                comm_y = _commission(qty_y_adjusted, py, self.cfg)
                slip_x = _slippage(qty_x, px, vx, self.cfg)
                slip_y = _slippage(qty_y_adjusted, py, vy, self.cfg)

                trade = Trade(
                    pair_id=pair_id,
                    entry_time=ts,
                    exit_time=None,
                    side=side,
                    entry_z=snap.z_score,
                    exit_z=None,
                    entry_price_x=px,
                    entry_price_y=py,
                    exit_price_x=None,
                    exit_price_y=None,
                    qty_x=qty_x,
                    qty_y=qty_y_adjusted,
                    commission=comm_x + comm_y,
                    slippage=slip_x + slip_y,
                )
                trade_id = str(uuid.uuid4())
                self._open_trades[pair_id] = (trade, trade_id)

            # ── HOLD: accrue borrow cost ───────────────────────────────────────
            elif snap.signal == SignalState.HOLD and pair_id in self._open_trades:
                trade, tid = self._open_trades[pair_id]
                # Short leg pays borrow cost
                borrow = _daily_borrow_cost(trade.qty_y, py, self.cfg)
                trade.commission += borrow  # fold into cost for simplicity

            # ── EXIT ───────────────────────────────────────────────────────────
            elif snap.signal == SignalState.EXIT and pair_id in self._open_trades:
                trade, tid = self._open_trades.pop(pair_id)

                # Exit execution costs
                comm_x = _commission(trade.qty_x, px, self.cfg)
                comm_y = _commission(trade.qty_y, py, self.cfg)
                slip_x = _slippage(trade.qty_x, px, vx, self.cfg)
                slip_y = _slippage(trade.qty_y, py, vy, self.cfg)
                trade.commission += comm_x + comm_y
                trade.slippage   += slip_x + slip_y

                # P&L computation
                # LONG  = long Y, short X  (spread expected to rise)
                # SHORT = short Y, long X  (spread expected to fall)
                if trade.side == Side.LONG:
                    pnl_y = (py - trade.entry_price_y) * trade.qty_y   # long Y
                    pnl_x = (trade.entry_price_x - px) * trade.qty_x   # short X
                else:
                    pnl_y = (trade.entry_price_y - py) * trade.qty_y   # short Y
                    pnl_x = (px - trade.entry_price_x) * trade.qty_x   # long X

                gross = pnl_x + pnl_y
                trade.exit_time    = ts
                trade.exit_z       = snap.z_score
                trade.exit_price_x = px
                trade.exit_price_y = py
                trade.gross_pnl    = gross
                trade.net_pnl      = gross - trade.commission - trade.slippage
                trade.is_open      = False

                completed.append(trade)

        # Force-close any still-open positions at last price
        for pair_id, (trade, tid) in list(self._open_trades.items()):
            trade.is_open = False
            completed.append(trade)
        self._open_trades.clear()

        return completed

    @property
    def open_trades(self) -> dict:
        return self._open_trades
