"""
StatArb Engine — Main entry point.

Usage:
    python main.py                   # synthetic data, full backtest + report
    python main.py --live            # attempt live yfinance download
    python main.py --capital 500000  # set starting capital
    python main.py --no-report       # skip HTML report generation
"""
import sys
import time
import argparse
from pathlib import Path
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine
from dashboard.report import build_dashboard
from execution.simulator import ExecutionConfig
from risk.manager import RiskConfig

console = Console()


def main():
    parser = argparse.ArgumentParser(description="StatArb Engine")
    parser.add_argument("--live",      action="store_true",
                        help="Use live yfinance data (requires internet)")
    parser.add_argument("--capital",   type=float, default=1_000_000.0,
                        help="Starting capital (default: $1,000,000)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML dashboard generation")
    parser.add_argument("--db",        type=str, default=":memory:",
                        help="DuckDB path (default: in-memory)")
    args = parser.parse_args()

    t0 = time.time()

    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════╗[/]")
    console.print("[bold cyan]║   STATISTICAL ARBITRAGE ENGINE       ║[/]")
    console.print("[bold cyan]║   Pairs Trading · Kalman · OU · VaR  ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/]")
    console.print()

    # Configure execution costs
    exec_cfg = ExecutionConfig(
        commission_per_share=0.005,
        min_commission=1.00,
        bid_ask_pct=0.001,
        market_impact_factor=0.1,
        borrow_rate_annual=0.005,
        capital_per_pair=args.capital / 5,
    )

    # Configure risk
    risk_cfg = RiskConfig(
        max_open_pairs=5,
        max_drawdown_pct=0.15,
        var_confidence=0.99,
    )

    # Run backtest
    engine = BacktestEngine(
        capital=args.capital,
        db_path=args.db,
        exec_cfg=exec_cfg,
        risk_cfg=risk_cfg,
    )
    results = engine.run(use_live_data=args.live)

    if not results:
        console.print("[red]Backtest produced no results.[/]")
        return

    elapsed = time.time() - t0
    console.print(f"\n[dim]Backtest completed in {elapsed:.2f}s[/]")

    # Generate HTML dashboard
    if not args.no_report:
        console.print("\n[yellow]Generating interactive dashboard...[/]")
        report_path = build_dashboard(
            results["db"],
            output_path="statarb_report.html",
        )
        console.print(f"  ✓ Report saved: [bold cyan]{report_path}[/]")
        console.print(f"    → Open in your browser to explore results")

    console.print()
    console.rule("[dim]Done[/]")


if __name__ == "__main__":
    main()
