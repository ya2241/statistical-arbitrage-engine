"""
DuckDB persistence layer.
All analytics queries are written in SQL — this is where the SQL requirement shines.
Uses a repository pattern so the rest of the codebase never constructs raw SQL.
"""
from __future__ import annotations
import duckdb
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from core.models import Trade, PortfolioSnapshot, SpreadSnapshot


DDL = """
CREATE TABLE IF NOT EXISTS prices (
    timestamp   TIMESTAMPTZ NOT NULL,
    ticker      VARCHAR     NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE      NOT NULL,
    volume      BIGINT,
    PRIMARY KEY (timestamp, ticker)
);

CREATE TABLE IF NOT EXISTS pairs (
    pair_id     VARCHAR PRIMARY KEY,
    ticker_x    VARCHAR NOT NULL,
    ticker_y    VARCHAR NOT NULL,
    hedge_ratio DOUBLE  NOT NULL,
    intercept   DOUBLE  NOT NULL,
    half_life   DOUBLE  NOT NULL,
    entry_z     DOUBLE  NOT NULL DEFAULT 2.0,
    exit_z      DOUBLE  NOT NULL DEFAULT 0.25,
    stop_z      DOUBLE  NOT NULL DEFAULT 3.5,
    lookback    INT     NOT NULL DEFAULT 60,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS spread_history (
    pair_id     VARCHAR     NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    spread      DOUBLE      NOT NULL,
    z_score     DOUBLE      NOT NULL,
    spread_mean DOUBLE      NOT NULL,
    spread_std  DOUBLE      NOT NULL,
    signal      VARCHAR     NOT NULL,
    PRIMARY KEY (pair_id, timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id        VARCHAR PRIMARY KEY,
    pair_id         VARCHAR     NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    side            VARCHAR     NOT NULL,
    entry_z         DOUBLE      NOT NULL,
    exit_z          DOUBLE,
    entry_price_x   DOUBLE      NOT NULL,
    entry_price_y   DOUBLE      NOT NULL,
    exit_price_x    DOUBLE,
    exit_price_y    DOUBLE,
    qty_x           DOUBLE      NOT NULL,
    qty_y           DOUBLE      NOT NULL,
    gross_pnl       DOUBLE      DEFAULT 0,
    commission      DOUBLE      DEFAULT 0,
    slippage        DOUBLE      DEFAULT 0,
    net_pnl         DOUBLE      DEFAULT 0,
    is_open         BOOLEAN     DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp           TIMESTAMPTZ PRIMARY KEY,
    nav                 DOUBLE NOT NULL,
    cash                DOUBLE NOT NULL,
    gross_exposure      DOUBLE NOT NULL,
    net_exposure        DOUBLE NOT NULL,
    num_open_trades     INT    NOT NULL,
    daily_pnl           DOUBLE NOT NULL,
    cumulative_pnl      DOUBLE NOT NULL,
    drawdown            DOUBLE NOT NULL,
    sharpe_rolling      DOUBLE DEFAULT 0
);
"""

ANALYTICS_QUERIES = {
    # --- Performance ---
    "total_return": """
        WITH ordered AS (
            SELECT nav, ROW_NUMBER() OVER (ORDER BY timestamp) AS rn,
                   COUNT(*) OVER () AS total_days
            FROM portfolio_snapshots
        )
        SELECT
            ROUND((MAX(nav) - MAX(CASE WHEN rn = 1 THEN nav END))
                  / NULLIF(MAX(CASE WHEN rn = 1 THEN nav END), 0) * 100, 2) AS total_return_pct,
            MAX(nav)    AS peak_nav,
            MIN(nav)    AS trough_nav,
            MAX(total_days) AS trading_days
        FROM ordered
    """,

    "monthly_pnl": """
        SELECT
            DATE_TRUNC('month', timestamp)  AS month,
            SUM(daily_pnl)                  AS monthly_pnl,
            COUNT(*)                        AS trading_days
        FROM portfolio_snapshots
        GROUP BY 1
        ORDER BY 1
    """,

    "pair_attribution": """
        SELECT
            pair_id,
            COUNT(*)                        AS num_trades,
            SUM(net_pnl)                    AS total_pnl,
            AVG(net_pnl)                    AS avg_pnl_per_trade,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS win_rate_pct,
            MAX(net_pnl)                    AS best_trade,
            MIN(net_pnl)                    AS worst_trade,
            AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))/86400) AS avg_hold_days
        FROM trades
        WHERE is_open = FALSE
        GROUP BY pair_id
        ORDER BY total_pnl DESC
    """,

    "drawdown_periods": """
        WITH nav_series AS (
            SELECT
                timestamp,
                nav,
                MAX(nav) OVER (ORDER BY timestamp ROWS UNBOUNDED PRECEDING) AS running_max
            FROM portfolio_snapshots
        )
        SELECT
            timestamp,
            nav,
            running_max,
            (nav - running_max) / running_max * 100 AS drawdown_pct
        FROM nav_series
        ORDER BY timestamp
    """,

    "rolling_sharpe": """
        WITH daily_returns AS (
            SELECT
                timestamp,
                daily_pnl / LAG(nav) OVER (ORDER BY timestamp) AS daily_ret
            FROM portfolio_snapshots
        )
        SELECT
            timestamp,
            AVG(daily_ret)   OVER w AS mean_ret,
            STDDEV(daily_ret) OVER w AS std_ret,
            CASE
                WHEN STDDEV(daily_ret) OVER w > 0
                THEN (AVG(daily_ret) OVER w / STDDEV(daily_ret) OVER w) * SQRT(252)
                ELSE 0
            END AS rolling_sharpe
        FROM daily_returns
        WINDOW w AS (ORDER BY timestamp ROWS 21 PRECEDING)
        ORDER BY timestamp
    """,

    "trade_duration_dist": """
        SELECT
            FLOOR(EXTRACT(EPOCH FROM (exit_time - entry_time))/3600) AS hold_hours,
            COUNT(*) AS num_trades,
            SUM(net_pnl) AS total_pnl
        FROM trades
        WHERE is_open = FALSE
        GROUP BY 1
        ORDER BY 1
    """,

    "z_score_analysis": """
        SELECT
            pair_id,
            ROUND(entry_z, 1)   AS entry_z_bucket,
            COUNT(*)            AS trades,
            AVG(net_pnl)        AS avg_pnl,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS win_rate
        FROM trades
        WHERE is_open = FALSE
        GROUP BY 1, 2
        ORDER BY 1, 2
    """,
}


class Database:
    def __init__(self, path: str = ":memory:"):
        self._path = path
        self._con = duckdb.connect(path)
        self._con.execute(DDL)

    def con(self) -> duckdb.DuckDBPyConnection:
        return self._con

    # ── Price data ────────────────────────────────────────────────────────────

    def upsert_prices(self, df: pd.DataFrame) -> None:
        """df must have columns: timestamp, ticker, open, high, low, close, volume"""
        self._con.execute("""
            INSERT OR REPLACE INTO prices
            SELECT timestamp, ticker, open, high, low, close, volume
            FROM df
        """)

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return self._con.execute("""
            SELECT timestamp, ticker, close
            FROM prices
            WHERE ticker = ANY(?)
              AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp, ticker
        """, [tickers, start, end]).df()

    # ── Spread history ────────────────────────────────────────────────────────

    def insert_spread_batch(self, pair_id: str, snaps: list[SpreadSnapshot]) -> None:
        rows = [(pair_id, s.timestamp, s.spread, s.z_score,
                 s.spread_mean, s.spread_std, s.signal.value) for s in snaps]
        self._con.executemany("""
            INSERT OR REPLACE INTO spread_history VALUES (?,?,?,?,?,?,?)
        """, rows)

    def get_spread_history(self, pair_id: str) -> pd.DataFrame:
        return self._con.execute("""
            SELECT * FROM spread_history WHERE pair_id = ? ORDER BY timestamp
        """, [pair_id]).df()

    # ── Trades ────────────────────────────────────────────────────────────────

    def upsert_trade(self, trade_id: str, t: Trade) -> None:
        self._con.execute("""
            INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [trade_id, t.pair_id, t.entry_time, t.exit_time, t.side.value,
              t.entry_z, t.exit_z, t.entry_price_x, t.entry_price_y,
              t.exit_price_x, t.exit_price_y, t.qty_x, t.qty_y,
              t.gross_pnl, t.commission, t.slippage, t.net_pnl, t.is_open])

    def get_closed_trades(self) -> pd.DataFrame:
        return self._con.execute("""
            SELECT * FROM trades WHERE is_open = FALSE ORDER BY exit_time
        """).df()

    # ── Portfolio snapshots ───────────────────────────────────────────────────

    def insert_snapshot(self, s: PortfolioSnapshot) -> None:
        self._con.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)
        """, [s.timestamp, s.nav, s.cash, s.gross_exposure, s.net_exposure,
              s.num_open_trades, s.daily_pnl, s.cumulative_pnl,
              s.drawdown, s.sharpe_rolling])

    def get_nav_series(self) -> pd.DataFrame:
        return self._con.execute("""
            SELECT timestamp, nav, cumulative_pnl, drawdown, sharpe_rolling
            FROM portfolio_snapshots ORDER BY timestamp
        """).df()

    # ── Analytics ─────────────────────────────────────────────────────────────

    def query(self, name: str) -> pd.DataFrame:
        sql = ANALYTICS_QUERIES.get(name)
        if sql is None:
            raise KeyError(f"Unknown query: {name!r}. Available: {list(ANALYTICS_QUERIES)}")
        return self._con.execute(sql).df()

    def raw(self, sql: str) -> pd.DataFrame:
        return self._con.execute(sql).df()
