"""
Event-driven backtest engine.

Architecture:
  - Iterates day-by-day in chronological order (no look-ahead)
  - For each day:
      1. Update spread state and generate signal
      2. Check risk limits before entry
      3. Simulate execution with realistic costs
      4. Update portfolio NAV and snapshot
  - Stores everything to DuckDB for post-trade analytics

This is the orchestrator — it wires all modules together.
"""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.progress import track
from rich.table import Table

from core.database import Database
from core.models import (PairConfig, SpreadSnapshot, SignalState,
                         Side, Trade, PortfolioSnapshot)
from data.fetcher import build_universe, fetch_live, LIVE_PAIRS
from strategy.cointegration import screen_pairs, CointResult
from strategy.signals import generate_signals
from execution.simulator import ExecutionSimulator, ExecutionConfig
from risk.manager import RiskManager, RiskConfig, compute_metrics

console = Console()


class BacktestEngine:
    """
    Statarb backtest engine.

    Usage:
        engine = BacktestEngine(capital=1_000_000)
        results = engine.run(use_live_data=False)
    """

    def __init__(
        self,
        capital: float = 1_000_000.0,
        db_path: str = ":memory:",
        exec_cfg: ExecutionConfig | None = None,
        risk_cfg: RiskConfig | None = None,
    ):
        self.capital = capital
        self.db = Database(db_path)
        self.exec_cfg = exec_cfg or ExecutionConfig(capital_per_pair=capital / 5)
        self.risk_cfg = risk_cfg or RiskConfig()

    def run(self, use_live_data: bool = False) -> dict:
        """
        Full backtest pipeline. Returns a dict of performance metrics.
        """
        console.rule("[bold cyan]StatArb Backtest Engine")

        # ── 1. Data ───────────────────────────────────────────────────────────
        console.print("[yellow]Loading market data...[/]")
        if use_live_data:
            all_tickers = [t for pair in LIVE_PAIRS for t in pair]
            prices = fetch_live(all_tickers)
            if prices is None:
                console.print("[red]Live data unavailable, falling back to synthetic[/]")
                use_live_data = False

        if not use_live_data:
            pair_list, prices = build_universe()
        else:
            pair_list = LIVE_PAIRS

        self.db.upsert_prices(prices)
        console.print(f"  ✓ {len(prices):,} price rows loaded for {prices['ticker'].nunique()} tickers")

        # ── 2. Cointegration screening ────────────────────────────────────────
        console.print("[yellow]Running cointegration screening...[/]")
        wide = prices.pivot_table(index="timestamp", columns="ticker", values="close")
        coint_results = screen_pairs(wide, pair_list)

        tradeable = [r for r in coint_results if r.is_cointegrated]
        console.print(f"  ✓ {len(tradeable)}/{len(pair_list)} pairs pass cointegration tests")

        self._print_pair_table(tradeable)

        if not tradeable:
            console.print("[red]No cointegrated pairs found. Exiting.[/]")
            return {}

        # ── 3. Build PairConfigs ──────────────────────────────────────────────
        pair_configs = [
            PairConfig(
                ticker_x=r.ticker_x,
                ticker_y=r.ticker_y,
                hedge_ratio=r.hedge_ratio,
                intercept=r.intercept,
                half_life_days=r.half_life_days,
            )
            for r in tradeable
        ]

        # ── 4. Generate signals for all pairs ─────────────────────────────────
        console.print("[yellow]Generating signals...[/]")
        pair_signals: dict[str, list[SpreadSnapshot]] = {}

        for pcfg in pair_configs:
            px = (wide[pcfg.ticker_x].dropna()
                  if pcfg.ticker_x in wide.columns else None)
            py = (wide[pcfg.ticker_y].dropna()
                  if pcfg.ticker_y in wide.columns else None)
            if px is None or py is None:
                continue
            idx = px.index.intersection(py.index)
            snaps = generate_signals(pcfg, px.loc[idx], py.loc[idx], use_kalman=True)
            pair_id = f"{pcfg.ticker_x}_{pcfg.ticker_y}"
            pair_signals[pair_id] = snaps
            self.db.insert_spread_batch(pair_id, snaps)

        console.print(f"  ✓ Signals generated for {len(pair_signals)} pairs")

        # ── 5. Event-driven simulation ────────────────────────────────────────
        console.print("[yellow]Running event-driven simulation...[/]")
        all_trades = self._simulate(pair_configs, pair_signals, wide)
        console.print(f"  ✓ {len(all_trades)} trades executed")

        # ── 6. Persist trades ─────────────────────────────────────────────────
        for trade in all_trades:
            self.db.upsert_trade(str(uuid.uuid4()), trade)

        # ── 7. Portfolio NAV series ───────────────────────────────────────────
        nav_series = self._build_nav_series(all_trades, wide)
        for snap in nav_series:
            self.db.insert_snapshot(snap)

        # ── 8. Compute metrics ────────────────────────────────────────────────
        daily_rets = np.diff([s.nav for s in nav_series]) / np.array([s.nav for s in nav_series[:-1]])
        metrics = compute_metrics(daily_rets)

        self._print_metrics(metrics)
        return {
            "metrics": metrics,
            "trades": all_trades,
            "nav_series": nav_series,
            "db": self.db,
            "pair_configs": pair_configs,
        }

    def _simulate(
        self,
        pair_configs: list[PairConfig],
        pair_signals: dict[str, list[SpreadSnapshot]],
        wide: pd.DataFrame,
    ) -> list[Trade]:
        """
        Per-pair execution simulation.
        """
        all_trades: list[Trade] = []

        for pcfg in pair_configs:
            pair_id = f"{pcfg.ticker_x}_{pcfg.ticker_y}"
            snaps = pair_signals.get(pair_id, [])
            if not snaps:
                continue

            px_series = wide.get(pcfg.ticker_x, pd.Series(dtype=float))
            py_series = wide.get(pcfg.ticker_y, pd.Series(dtype=float))

            prices_x = dict(zip(px_series.index, px_series.values))
            prices_y = dict(zip(py_series.index, py_series.values))

            sim = ExecutionSimulator(self.exec_cfg)
            trades = sim.process(
                pair_cfg=pcfg,
                snapshots=snaps,
                prices_x=prices_x,
                prices_y=prices_y,
                volumes_x={k: 5_000_000 for k in prices_x},
                volumes_y={k: 5_000_000 for k in prices_y},
            )
            all_trades.extend(trades)

        return all_trades

    def _build_nav_series(
        self,
        trades: list[Trade],
        wide: pd.DataFrame,
    ) -> list[PortfolioSnapshot]:
        """
        Reconstruct daily NAV from closed trades.
        Assumes capital_per_pair is recycled after each trade.
        """
        if not trades:
            return []

        # Build a daily P&L series
        dates = sorted(wide.index)
        daily_pnl: dict = {d: 0.0 for d in dates}

        for t in trades:
            if t.exit_time and not t.is_open:
                exit_date = pd.Timestamp(t.exit_time).normalize()
                if exit_date in daily_pnl:
                    daily_pnl[exit_date] += t.net_pnl

        nav = self.capital
        peak_nav = nav
        cumulative_pnl = 0.0
        snapshots: list[PortfolioSnapshot] = []
        recent_returns: list[float] = []

        for dt in dates:
            pnl = daily_pnl.get(dt, 0.0)
            prev_nav = nav
            nav += pnl
            cumulative_pnl += pnl
            peak_nav = max(peak_nav, nav)
            drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

            daily_ret = pnl / prev_nav if prev_nav > 0 else 0.0
            recent_returns.append(daily_ret)

            # Rolling 21-day Sharpe
            if len(recent_returns) >= 21:
                r21 = np.array(recent_returns[-21:])
                mu, sigma = np.mean(r21), np.std(r21, ddof=1)
                sharpe_21 = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0
            else:
                sharpe_21 = 0.0

            open_count = sum(1 for t in trades if t.is_open)
            snapshots.append(PortfolioSnapshot(
                timestamp=dt,
                nav=nav,
                cash=nav,
                gross_exposure=self.exec_cfg.capital_per_pair * open_count * 2,
                net_exposure=0.0,
                num_open_trades=open_count,
                daily_pnl=pnl,
                cumulative_pnl=cumulative_pnl,
                drawdown=drawdown,
                sharpe_rolling=sharpe_21,
            ))

        return snapshots

    def _print_pair_table(self, results: list[CointResult]) -> None:
        t = Table(title="Cointegrated Pairs", show_header=True,
                  header_style="bold magenta")
        t.add_column("Pair",        style="cyan",   width=14)
        t.add_column("EG p-val",    justify="right", width=10)
        t.add_column("ADF p-val",   justify="right", width=10)
        t.add_column("Half-life",   justify="right", width=10)
        t.add_column("Hedge Ratio", justify="right", width=12)
        t.add_column("Score",       justify="right", width=8)

        for r in results:
            pair = f"{r.ticker_x}/{r.ticker_y}"
            t.add_row(pair, f"{r.eg_pvalue:.4f}", f"{r.adf_pvalue:.4f}",
                      f"{r.half_life_days:.1f}d", f"{r.hedge_ratio:.4f}",
                      f"{r.score:.3f}")
        console.print(t)

    def _print_metrics(self, m: dict) -> None:
        if not m:
            return
        t = Table(title="Backtest Performance", show_header=True,
                  header_style="bold green")
        t.add_column("Metric", style="cyan", width=28)
        t.add_column("Value",  justify="right", width=14)

        rows = [
            ("Total Return",        f"{m.get('total_return_pct',0):.2f}%"),
            ("Annual Return",       f"{m.get('annual_return_pct',0):.2f}%"),
            ("Annual Volatility",   f"{m.get('annual_vol_pct',0):.2f}%"),
            ("Sharpe Ratio",        f"{m.get('sharpe_ratio',0):.3f}"),
            ("Sortino Ratio",       f"{m.get('sortino_ratio',0):.3f}"),
            ("Calmar Ratio",        f"{m.get('calmar_ratio',0):.3f}"),
            ("Max Drawdown",        f"{m.get('max_drawdown_pct',0):.2f}%"),
            ("Win Rate",            f"{m.get('win_rate_pct',0):.1f}%"),
            ("Skewness",            f"{m.get('skewness',0):.3f}"),
            ("Excess Kurtosis",     f"{m.get('excess_kurtosis',0):.3f}"),
            ("VaR 95%",             f"{m.get('var_95_pct',0):.3f}%"),
            ("CVaR 95%",            f"{m.get('cvar_95_pct',0):.3f}%"),
        ]
        for label, val in rows:
            t.add_row(label, val)
        console.print(t)
