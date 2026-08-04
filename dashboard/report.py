"""
Interactive HTML dashboard using Plotly.
Generates a self-contained single-file report — no server required.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from core.database import Database


DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
CYAN      = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
PURPLE    = "#bc8cff"
ORANGE    = "#ffa657"


def _base_layout(title: str = "") -> dict:
    return dict(
        title=title,
        plot_bgcolor=DARK_BG,
        paper_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COL, family="JetBrains Mono, monospace", size=11),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COL, borderwidth=1),
    )


def build_dashboard(db: Database, output_path: str = "statarb_report.html") -> str:
    """
    Build a comprehensive interactive HTML report from the database.
    Returns the path to the generated file.
    """

    # ── Pull data from DuckDB ──────────────────────────────────────────────────
    nav_df      = db.get_nav_series()
    trades_df   = db.get_closed_trades()
    attr_df     = db.query("pair_attribution")
    monthly_df  = db.query("monthly_pnl")
    drawdown_df = db.query("drawdown_periods")
    sharpe_df   = db.query("rolling_sharpe")

    # ── Figure 1: NAV + Drawdown ───────────────────────────────────────────────
    fig_nav = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )

    if not nav_df.empty:
        fig_nav.add_trace(go.Scatter(
            x=nav_df["timestamp"], y=nav_df["nav"],
            mode="lines", name="NAV",
            line=dict(color=CYAN, width=2),
            fill="tonexty", fillcolor=f"rgba(88,166,255,0.08)",
        ), row=1, col=1)

        if not drawdown_df.empty:
            fig_nav.add_trace(go.Scatter(
                x=drawdown_df["timestamp"],
                y=drawdown_df["drawdown_pct"],
                mode="lines", name="Drawdown %",
                line=dict(color=RED, width=1.5),
                fill="tozeroy", fillcolor=f"rgba(248,81,73,0.15)",
            ), row=2, col=1)

    fig_nav.update_layout(**_base_layout("Portfolio NAV & Drawdown"))
    fig_nav.update_yaxes(title_text="NAV ($)", row=1)
    fig_nav.update_yaxes(title_text="Drawdown %", row=2)

    # ── Figure 2: Monthly P&L bar chart ───────────────────────────────────────
    fig_monthly = go.Figure()
    if not monthly_df.empty:
        colours = [GREEN if v >= 0 else RED for v in monthly_df["monthly_pnl"]]
        fig_monthly.add_trace(go.Bar(
            x=monthly_df["month"].astype(str),
            y=monthly_df["monthly_pnl"],
            marker_color=colours,
            name="Monthly P&L",
            text=[f"${v:,.0f}" for v in monthly_df["monthly_pnl"]],
            textposition="outside",
        ))
    fig_monthly.update_layout(**_base_layout("Monthly P&L Attribution"))
    fig_monthly.update_yaxes(title_text="P&L ($)")

    # ── Figure 3: Pair attribution horizontal bar ──────────────────────────────
    fig_pairs = go.Figure()
    if not attr_df.empty:
        attr_sorted = attr_df.sort_values("total_pnl")
        fig_pairs.add_trace(go.Bar(
            y=attr_sorted["pair_id"],
            x=attr_sorted["total_pnl"],
            orientation="h",
            marker_color=[GREEN if v >= 0 else RED for v in attr_sorted["total_pnl"]],
            text=[f"${v:,.0f}" for v in attr_sorted["total_pnl"]],
            textposition="outside",
        ))
    fig_pairs.update_layout(**_base_layout("P&L by Pair"))
    fig_pairs.update_xaxes(title_text="Total Net P&L ($)")

    # ── Figure 4: Rolling Sharpe ───────────────────────────────────────────────
    fig_sharpe = go.Figure()
    if not sharpe_df.empty and "rolling_sharpe" in sharpe_df.columns:
        sharpe_clean = sharpe_df.dropna(subset=["rolling_sharpe"])
        fig_sharpe.add_trace(go.Scatter(
            x=sharpe_clean["timestamp"],
            y=sharpe_clean["rolling_sharpe"],
            mode="lines", name="21D Rolling Sharpe",
            line=dict(color=PURPLE, width=1.5),
        ))
        fig_sharpe.add_hline(y=1.0, line_dash="dash",
                             line_color=YELLOW, annotation_text="SR=1")
        fig_sharpe.add_hline(y=0.0, line_dash="dot",
                             line_color=GRID_COL)
    fig_sharpe.update_layout(**_base_layout("21-Day Rolling Sharpe Ratio"))
    fig_sharpe.update_yaxes(title_text="Sharpe Ratio")

    # ── Figure 5: Trade P&L scatter ───────────────────────────────────────────
    fig_scatter = go.Figure()
    if not trades_df.empty and "entry_z" in trades_df.columns:
        colours_t = [GREEN if v > 0 else RED for v in trades_df["net_pnl"]]
        fig_scatter.add_trace(go.Scatter(
            x=trades_df["entry_z"],
            y=trades_df["net_pnl"],
            mode="markers",
            marker=dict(
                color=colours_t,
                size=8,
                opacity=0.7,
                line=dict(color=GRID_COL, width=0.5),
            ),
            text=trades_df["pair_id"],
            name="Trades",
        ))
        fig_scatter.add_hline(y=0, line_dash="dot", line_color=GRID_COL)
    fig_scatter.update_layout(**_base_layout("Trade P&L vs Entry Z-Score"))
    fig_scatter.update_xaxes(title_text="Entry Z-Score")
    fig_scatter.update_yaxes(title_text="Net P&L ($)")

    # ── Figure 6: Win rate by pair ─────────────────────────────────────────────
    fig_winrate = go.Figure()
    if not attr_df.empty and "win_rate_pct" in attr_df.columns:
        fig_winrate.add_trace(go.Bar(
            x=attr_df["pair_id"],
            y=attr_df["win_rate_pct"],
            marker_color=[GREEN if v >= 50 else RED for v in attr_df["win_rate_pct"]],
            text=[f"{v:.1f}%" for v in attr_df["win_rate_pct"]],
            textposition="outside",
        ))
        fig_winrate.add_hline(y=50, line_dash="dash", line_color=YELLOW,
                              annotation_text="50%")
    fig_winrate.update_layout(**_base_layout("Win Rate by Pair"))
    fig_winrate.update_yaxes(title_text="Win Rate (%)", range=[0, 110])

    # ── Metrics summary table ──────────────────────────────────────────────────
    total_row = db.query("total_return")

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = _render_html(
        nav_fig=fig_nav,
        monthly_fig=fig_monthly,
        pairs_fig=fig_pairs,
        sharpe_fig=fig_sharpe,
        scatter_fig=fig_scatter,
        winrate_fig=fig_winrate,
        attr_df=attr_df,
        total_row=total_row,
        trades_df=trades_df,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _render_html(nav_fig, monthly_fig, pairs_fig, sharpe_fig,
                 scatter_fig, winrate_fig, attr_df, total_row, trades_df) -> str:
    """Render all figures into a single polished HTML page."""

    def fig_html(fig) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    # Compute headline stats
    total_pnl   = trades_df["net_pnl"].sum() if not trades_df.empty else 0
    num_trades  = len(trades_df)
    win_rate    = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0
    avg_pnl     = trades_df["net_pnl"].mean() if not trades_df.empty else 0
    total_comm  = trades_df["commission"].sum() if not trades_df.empty else 0
    total_slip  = trades_df["slippage"].sum() if not trades_df.empty else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StatArb Engine — Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:     {DARK_BG};
    --panel:  {PANEL_BG};
    --border: {GRID_COL};
    --text:   {TEXT_COL};
    --cyan:   {CYAN};
    --green:  {GREEN};
    --red:    {RED};
    --yellow: {YELLOW};
    --purple: {PURPLE};
  }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    line-height: 1.6;
  }}

  /* ── Header ── */
  .header {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 24px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  .header-logo {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: -0.5px;
  }}
  .header-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--yellow);
    background: rgba(210,153,34,0.12);
    border: 1px solid rgba(210,153,34,0.3);
    padding: 2px 10px;
    border-radius: 4px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }}
  .header-subtitle {{
    margin-left: auto;
    font-size: 11px;
    color: #7d8590;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* ── KPI cards ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .kpi-card {{
    background: var(--panel);
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .kpi-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7d8590;
    font-family: 'JetBrains Mono', monospace;
  }}
  .kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
  }}
  .kpi-sub {{
    font-size: 10px;
    color: #7d8590;
  }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .neu {{ color: var(--cyan); }}
  .warn {{ color: var(--yellow); }}

  /* ── Main layout ── */
  .main {{
    padding: 32px 40px;
    display: flex;
    flex-direction: column;
    gap: 24px;
    max-width: 1800px;
  }}
  .section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #7d8590;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .chart-grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  .chart-grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
  }}
  .chart-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    overflow: hidden;
  }}
  .chart-card-full {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    overflow: hidden;
  }}

  /* ── Attribution table ── */
  .attr-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }}
  .attr-table th {{
    background: var(--bg);
    padding: 10px 16px;
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #7d8590;
    border-bottom: 1px solid var(--border);
  }}
  .attr-table td {{
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  .attr-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

  /* ── Footer ── */
  .footer {{
    margin: 32px 40px;
    padding: 20px;
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #7d8590;
    display: flex;
    justify-content: space-between;
  }}

  /* ── Pill badge ── */
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
  }}
  .badge-green  {{ background: rgba(63,185,80,0.15);  color: var(--green); }}
  .badge-red    {{ background: rgba(248,81,73,0.15);   color: var(--red);  }}
  .badge-cyan   {{ background: rgba(88,166,255,0.15); color: var(--cyan); }}
</style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────────────── -->
<div class="header">
  <div class="header-logo">⟁ STATARB</div>
  <div class="header-tag">Backtest Report</div>
  <div class="header-subtitle">Statistical Arbitrage Engine · Python + DuckDB · Kalman Filter Hedge Ratio</div>
</div>

<!-- ── KPI Strip ──────────────────────────────────────────────────────────── -->
<div class="kpi-grid">
  <div class="kpi-card">
    <span class="kpi-label">Net P&amp;L</span>
    <span class="kpi-value {'pos' if total_pnl >= 0 else 'neg'}">${total_pnl:,.0f}</span>
    <span class="kpi-sub">After all costs</span>
  </div>
  <div class="kpi-card">
    <span class="kpi-label">Total Trades</span>
    <span class="kpi-value neu">{num_trades}</span>
    <span class="kpi-sub">Round-trip</span>
  </div>
  <div class="kpi-card">
    <span class="kpi-label">Win Rate</span>
    <span class="kpi-value {'pos' if win_rate >= 50 else 'neg'}">{win_rate:.1f}%</span>
    <span class="kpi-sub">By trade count</span>
  </div>
  <div class="kpi-card">
    <span class="kpi-label">Avg P&amp;L / Trade</span>
    <span class="kpi-value {'pos' if avg_pnl >= 0 else 'neg'}">${avg_pnl:,.0f}</span>
    <span class="kpi-sub">Net of costs</span>
  </div>
  <div class="kpi-card">
    <span class="kpi-label">Total Commission</span>
    <span class="kpi-value warn">${total_comm:,.0f}</span>
    <span class="kpi-sub">Entry + exit + borrow</span>
  </div>
  <div class="kpi-card">
    <span class="kpi-label">Total Slippage</span>
    <span class="kpi-value warn">${total_slip:,.0f}</span>
    <span class="kpi-sub">Impact + spread</span>
  </div>
</div>

<!-- ── Main content ───────────────────────────────────────────────────────── -->
<div class="main">

  <!-- NAV + Drawdown (full width) -->
  <div>
    <div class="section-title">Portfolio Performance</div>
    <div class="chart-card-full">
      {fig_html(nav_fig)}
    </div>
  </div>

  <!-- Monthly PnL + Rolling Sharpe -->
  <div>
    <div class="section-title">Return Profile</div>
    <div class="chart-grid-2">
      <div class="chart-card">{fig_html(monthly_fig)}</div>
      <div class="chart-card">{fig_html(sharpe_fig)}</div>
    </div>
  </div>

  <!-- Pair attribution + Win rate + Trade scatter -->
  <div>
    <div class="section-title">Pair Analytics</div>
    <div class="chart-grid-3">
      <div class="chart-card">{fig_html(pairs_fig)}</div>
      <div class="chart-card">{fig_html(winrate_fig)}</div>
      <div class="chart-card">{fig_html(scatter_fig)}</div>
    </div>
  </div>

  <!-- Attribution table -->
  <div>
    <div class="section-title">Pair Attribution Detail</div>
    <div class="chart-card-full">
      <table class="attr-table">
        <thead>
          <tr>
            <th>Pair</th>
            <th>Trades</th>
            <th>Total P&L</th>
            <th>Avg P&L</th>
            <th>Win Rate</th>
            <th>Best Trade</th>
            <th>Worst Trade</th>
            <th>Avg Hold</th>
          </tr>
        </thead>
        <tbody>
          {''.join(
            f"""<tr>
              <td><span class="badge badge-cyan">{row.pair_id}</span></td>
              <td>{int(row.num_trades)}</td>
              <td class="{'pos' if row.total_pnl >= 0 else 'neg'}">${row.total_pnl:,.0f}</td>
              <td class="{'pos' if row.avg_pnl_per_trade >= 0 else 'neg'}">${row.avg_pnl_per_trade:,.0f}</td>
              <td><span class="badge {'badge-green' if row.win_rate_pct >= 50 else 'badge-red'}">{row.win_rate_pct:.1f}%</span></td>
              <td class="pos">${row.best_trade:,.0f}</td>
              <td class="neg">${row.worst_trade:,.0f}</td>
              <td>{row.avg_hold_days:.1f}d</td>
            </tr>"""
            for row in attr_df.itertuples()
          ) if not attr_df.empty else '<tr><td colspan="8" style="text-align:center;color:#7d8590">No data</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>

</div>

<!-- ── Footer ─────────────────────────────────────────────────────────────── -->
<div class="footer">
  <span>StatArb Engine · Cointegration + Kalman Filter + OU Half-Life + Parametric VaR · Python + DuckDB</span>
  <span>Costs: Commission $0.005/share · Bid-ask 10bps · √ADV market impact · Short borrow 50bps/yr</span>
</div>

</body>
</html>"""
