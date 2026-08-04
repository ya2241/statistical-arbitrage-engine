-- ============================================================
-- StatArb Engine — Analytics SQL Showcase
-- Engine: DuckDB (analytical SQL with window functions)
-- ============================================================
-- These queries are designed to run directly against the
-- DuckDB database produced by the backtest engine.
-- They demonstrate production-quality SQL: window functions,
-- CTEs, conditional aggregations, and time-series arithmetic.
-- ============================================================


-- ── 1. Performance Summary ────────────────────────────────────────────────────
-- Compute total return, peak NAV, and trading days from snapshots.

SELECT
    ROUND((MAX(nav) - FIRST(nav)) / FIRST(nav) * 100, 2)   AS total_return_pct,
    ROUND(MAX(nav), 2)                                       AS peak_nav,
    ROUND(MIN(nav), 2)                                       AS trough_nav,
    ROUND(MAX(nav) - MIN(nav), 2)                            AS nav_range,
    COUNT(*)                                                 AS trading_days
FROM portfolio_snapshots
ORDER BY timestamp;


-- ── 2. Monthly P&L Waterfall ──────────────────────────────────────────────────
-- Break down cumulative P&L month by month.

SELECT
    STRFTIME(timestamp, '%Y-%m')      AS month,
    SUM(daily_pnl)                    AS monthly_pnl,
    SUM(SUM(daily_pnl)) OVER (
        ORDER BY DATE_TRUNC('month', timestamp)
        ROWS UNBOUNDED PRECEDING
    )                                 AS cumulative_pnl,
    COUNT(*)                          AS trading_days,
    ROUND(SUM(daily_pnl) / COUNT(*), 2) AS avg_daily_pnl
FROM portfolio_snapshots
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY DATE_TRUNC('month', timestamp);


-- ── 3. Drawdown Analysis ──────────────────────────────────────────────────────
-- Find all drawdown periods with depth and duration.

WITH nav_series AS (
    SELECT
        timestamp,
        nav,
        MAX(nav) OVER (
            ORDER BY timestamp
            ROWS UNBOUNDED PRECEDING
        ) AS running_peak
    FROM portfolio_snapshots
),
drawdown_flags AS (
    SELECT
        timestamp,
        nav,
        running_peak,
        (nav - running_peak) / running_peak * 100 AS drawdown_pct,
        CASE WHEN nav < running_peak THEN 1 ELSE 0 END AS in_drawdown
    FROM nav_series
),
drawdown_groups AS (
    SELECT *,
        SUM(CASE WHEN in_drawdown = 0 THEN 1 ELSE 0 END)
            OVER (ORDER BY timestamp) AS dd_group
    FROM drawdown_flags
)
SELECT
    MIN(timestamp)          AS drawdown_start,
    MAX(timestamp)          AS drawdown_end,
    MIN(drawdown_pct)       AS max_depth_pct,
    COUNT(*)                AS duration_days
FROM drawdown_groups
WHERE in_drawdown = 1
GROUP BY dd_group
ORDER BY max_depth_pct;


-- ── 4. Rolling Sharpe (21-Day) ────────────────────────────────────────────────
-- Annualised Sharpe ratio on a rolling 21-day window.

WITH daily_returns AS (
    SELECT
        timestamp,
        daily_pnl / NULLIF(LAG(nav) OVER (ORDER BY timestamp), 0) AS daily_ret
    FROM portfolio_snapshots
)
SELECT
    timestamp,
    ROUND(AVG(daily_ret) OVER w, 6)      AS mean_ret,
    ROUND(STDDEV(daily_ret) OVER w, 6)   AS std_ret,
    ROUND(
        CASE
            WHEN STDDEV(daily_ret) OVER w > 0
            THEN AVG(daily_ret) OVER w
                 / STDDEV(daily_ret) OVER w
                 * SQRT(252)
            ELSE 0
        END, 3
    )                                     AS rolling_sharpe_21d
FROM daily_returns
WINDOW w AS (ORDER BY timestamp ROWS 20 PRECEDING)
ORDER BY timestamp;


-- ── 5. Pair Attribution with Statistical Summary ──────────────────────────────
-- Full per-pair breakdown: P&L, efficiency, win/loss anatomy.

SELECT
    pair_id,
    COUNT(*)                                                   AS num_trades,
    ROUND(SUM(net_pnl), 2)                                    AS total_net_pnl,
    ROUND(SUM(gross_pnl), 2)                                  AS total_gross_pnl,
    ROUND(SUM(commission), 2)                                  AS total_commission,
    ROUND(SUM(slippage), 2)                                    AS total_slippage,
    ROUND(AVG(net_pnl), 2)                                    AS avg_pnl_per_trade,
    ROUND(STDDEV(net_pnl), 2)                                 AS pnl_std,
    -- Win rate
    ROUND(
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                          AS win_rate_pct,
    -- Profit factor: gross_win / gross_loss
    ROUND(
        SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0 END)), 0)
    , 2)                                                       AS profit_factor,
    ROUND(MAX(net_pnl), 2)                                    AS best_trade,
    ROUND(MIN(net_pnl), 2)                                    AS worst_trade,
    -- Average hold time in days
    ROUND(
        AVG(EXTRACT(EPOCH FROM (exit_time - entry_time)) / 86400.0)
    , 1)                                                       AS avg_hold_days,
    -- Cost drag as % of gross P&L
    ROUND(
        (SUM(commission) + SUM(slippage))
        / NULLIF(ABS(SUM(gross_pnl)), 0) * 100, 1
    )                                                          AS cost_drag_pct
FROM trades
WHERE is_open = FALSE
GROUP BY pair_id
ORDER BY total_net_pnl DESC;


-- ── 6. Z-Score Entry Analysis ─────────────────────────────────────────────────
-- Does entry z-score predict trade profitability?

SELECT
    pair_id,
    ROUND(entry_z, 1)                                          AS entry_z_bucket,
    side,
    COUNT(*)                                                   AS trades,
    ROUND(AVG(net_pnl), 2)                                    AS avg_pnl,
    ROUND(AVG(ABS(entry_z) - ABS(COALESCE(exit_z, 0))), 3)   AS avg_z_reversion,
    ROUND(
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                          AS win_rate_pct
FROM trades
WHERE is_open = FALSE
GROUP BY pair_id, ROUND(entry_z, 1), side
ORDER BY pair_id, entry_z_bucket;


-- ── 7. Cost Decomposition ─────────────────────────────────────────────────────
-- Understand where transaction costs go.

SELECT
    pair_id,
    COUNT(*)                                      AS trades,
    ROUND(SUM(gross_pnl), 2)                     AS gross_pnl,
    ROUND(SUM(commission), 2)                     AS commission,
    ROUND(SUM(slippage), 2)                       AS slippage,
    ROUND(SUM(net_pnl), 2)                        AS net_pnl,
    ROUND(SUM(commission) / NULLIF(SUM(gross_pnl), 0) * 100, 2) AS comm_pct_of_gross,
    ROUND(SUM(slippage)   / NULLIF(SUM(gross_pnl), 0) * 100, 2) AS slip_pct_of_gross
FROM trades
WHERE is_open = FALSE
GROUP BY pair_id
ORDER BY net_pnl DESC;


-- ── 8. Trade Duration Distribution ───────────────────────────────────────────
-- How long do we hold positions, and does it correlate with profit?

SELECT
    CASE
        WHEN EXTRACT(EPOCH FROM (exit_time - entry_time)) / 86400 < 3   THEN '0-3 days'
        WHEN EXTRACT(EPOCH FROM (exit_time - entry_time)) / 86400 < 7   THEN '3-7 days'
        WHEN EXTRACT(EPOCH FROM (exit_time - entry_time)) / 86400 < 14  THEN '1-2 weeks'
        WHEN EXTRACT(EPOCH FROM (exit_time - entry_time)) / 86400 < 30  THEN '2-4 weeks'
        ELSE '1+ month'
    END                              AS hold_bucket,
    COUNT(*)                         AS num_trades,
    ROUND(AVG(net_pnl), 2)          AS avg_pnl,
    ROUND(SUM(net_pnl), 2)          AS total_pnl,
    ROUND(
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                AS win_rate_pct
FROM trades
WHERE is_open = FALSE
GROUP BY hold_bucket
ORDER BY MIN(EXTRACT(EPOCH FROM (exit_time - entry_time)));


-- ── 9. Consecutive Win/Loss Streaks ──────────────────────────────────────────
-- Identify the longest winning and losing streaks per pair.

WITH ranked AS (
    SELECT
        pair_id,
        exit_time,
        net_pnl,
        SIGN(net_pnl)                            AS outcome,
        ROW_NUMBER() OVER (ORDER BY exit_time)   AS rn
    FROM trades
    WHERE is_open = FALSE
),
streak_groups AS (
    SELECT *,
        rn - ROW_NUMBER() OVER (
            PARTITION BY pair_id, SIGN(net_pnl)
            ORDER BY exit_time
        ) AS grp
    FROM ranked
)
SELECT
    pair_id,
    CASE WHEN outcome = 1 THEN 'Win' ELSE 'Loss' END AS streak_type,
    COUNT(*)                                           AS streak_length,
    ROUND(SUM(net_pnl), 2)                            AS streak_pnl,
    MIN(exit_time)                                     AS streak_start,
    MAX(exit_time)                                     AS streak_end
FROM streak_groups
GROUP BY pair_id, outcome, grp
ORDER BY streak_length DESC
LIMIT 20;


-- ── 10. Spread Mean-Reversion Speed by Pair ───────────────────────────────────
-- Verify that signal z-scores are reverting as expected.

SELECT
    pair_id,
    ROUND(AVG(ABS(z_score)), 3)              AS avg_abs_z,
    ROUND(MAX(ABS(z_score)), 3)              AS max_abs_z,
    ROUND(STDDEV(z_score), 3)                AS z_std,
    ROUND(AVG(CASE WHEN signal = 'HOLD' THEN ABS(z_score) END), 3)  AS avg_z_while_holding,
    COUNT(CASE WHEN signal = 'ENTER' THEN 1 END)                      AS num_entries,
    COUNT(CASE WHEN signal = 'EXIT'  THEN 1 END)                      AS num_exits
FROM spread_history
GROUP BY pair_id
ORDER BY pair_id;
