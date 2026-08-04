"""
Cointegration research engine.

Pipeline:
  1. Engle-Granger two-step test on each candidate pair
  2. Johansen trace test for robustness
  3. OLS hedge ratio with Kalman-filtered dynamic hedge ratio (advanced)
  4. Ornstein-Uhlenbeck half-life estimation via MLE
  5. Pair scoring: combine statistical strength + half-life suitability
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class CointResult:
    ticker_x: str
    ticker_y: str
    eg_pvalue: float          # Engle-Granger p-value
    johansen_trace_stat: float
    johansen_critical_5: float
    hedge_ratio: float        # OLS beta
    intercept: float
    half_life_days: float     # OU half-life
    adf_pvalue: float         # ADF on residuals
    spread_mean: float
    spread_std: float
    score: float = 0.0        # composite attractiveness score

    @property
    def is_cointegrated(self) -> bool:
        return (self.eg_pvalue < 0.05
                and self.adf_pvalue < 0.05
                and self.johansen_trace_stat > self.johansen_critical_5
                and 5 <= self.half_life_days <= 60)


def _estimate_ou_half_life(spread: np.ndarray) -> float:
    """
    Estimate Ornstein-Uhlenbeck half-life via linear regression:
      Δspread_t = α + β * spread_{t-1} + ε
    Half-life = -ln(2) / β
    """
    spread = np.asarray(spread, dtype=float)
    delta  = np.diff(spread)
    lag    = spread[:-1]
    # OLS: delta ~ lag
    b, a, *_ = np.polyfit(lag, delta, 1)
    if b >= 0:
        return np.inf  # not mean-reverting
    return -np.log(2) / b


def _ou_half_life_mle(spread: np.ndarray, dt: float = 1.0) -> float:
    """
    More accurate OU parameter estimation using exact discrete-time MLE.
    Uses the fact that the exact transition density is Gaussian.
    dt=1.0 means each step is 1 trading day → half-life returned in days.
    """
    s = np.asarray(spread, dtype=float)
    n = len(s) - 1
    sx  = np.sum(s[:-1])
    sy  = np.sum(s[1:])
    sxx = np.sum(s[:-1]**2)
    sxy = np.sum(s[:-1] * s[1:])

    denom = n * sxx - sx**2
    if abs(denom) < 1e-10:
        return _estimate_ou_half_life(spread)

    # Slope of s[t] on s[t-1]: if < 1, process is mean-reverting
    beta = (n * sxy - sx * sy) / denom
    if beta >= 1.0 or beta <= 0.0:
        return _estimate_ou_half_life(spread)

    # theta is the daily mean-reversion rate: beta = exp(-theta * dt)
    theta = -np.log(beta) / dt
    if theta <= 0:
        return _estimate_ou_half_life(spread)

    return np.log(2) / theta


def run_engle_granger(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float, float, float, float]:
    """
    OLS regression of y on x, then ADF test on residuals.
    Returns: (p_value, hedge_ratio, intercept, adf_pvalue, residual_std)
    """
    x_arr = np.asarray(x, float)
    y_arr = np.asarray(y, float)

    # OLS with intercept
    X = np.column_stack([x_arr, np.ones(len(x_arr))])
    coeffs, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    hedge_ratio, intercept = coeffs

    residuals = y_arr - (hedge_ratio * x_arr + intercept)

    # Engle-Granger test
    _, pval, _ = coint(x_arr, y_arr, trend="c")

    # ADF on residuals to confirm stationarity
    adf_stat, adf_pval, *_ = adfuller(residuals, autolag="AIC")

    return pval, hedge_ratio, intercept, adf_pval, float(np.std(residuals))


def run_johansen(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Johansen trace test. Returns (trace_statistic, 5pct_critical_value).
    """
    data = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        return float(result.lr1[0]), float(result.cvt[0, 1])  # trace, 5% CV
    except Exception:
        return 0.0, 999.0


def score_pair(r: CointResult) -> float:
    """
    Composite attractiveness score for a cointegrated pair.
    Higher = more attractive for trading.

    Components:
      - Statistical confidence (lower p-values → higher score)
      - Half-life in sweet spot (10-25 days ideal for daily strategy)
      - Spread volatility (higher → more opportunity)
    """
    if not r.is_cointegrated:
        return 0.0

    # P-value component: -log(p), capped
    stat_score = min(-np.log10(r.eg_pvalue + 1e-10), 4.0) / 4.0

    # Half-life component: Gaussian centred at 18 days, width 10
    hl_score = np.exp(-((r.half_life_days - 18) ** 2) / (2 * 10**2))

    # Volatility component: more spread vol = more profit opportunity
    vol_score = min(r.spread_std / 3.0, 1.0)

    return 0.5 * stat_score + 0.3 * hl_score + 0.2 * vol_score


def screen_pairs(
    prices: pd.DataFrame,
    candidates: list[tuple[str, str]],
    min_obs: int = 252,
) -> list[CointResult]:
    """
    Screen a list of candidate pairs for cointegration.

    Args:
        prices:     Wide-format close prices, columns are tickers.
        candidates: List of (ticker_x, ticker_y) tuples.
        min_obs:    Minimum number of observations required.

    Returns:
        List of CointResult sorted by score descending.
    """
    # Pivot to wide
    if "ticker" in prices.columns:
        wide = prices.pivot_table(
            index="timestamp", columns="ticker", values="close"
        ).sort_index()
    else:
        wide = prices.sort_index()

    results: list[CointResult] = []

    for tx, ty in candidates:
        if tx not in wide.columns or ty not in wide.columns:
            continue

        pair = wide[[tx, ty]].dropna()
        if len(pair) < min_obs:
            continue

        x, y = pair[tx], pair[ty]

        # Engle-Granger
        eg_pval, hr, intercept, adf_pval, res_std = run_engle_granger(x, y)

        # Johansen
        joh_trace, joh_cv5 = run_johansen(x, y)

        # Spread for half-life
        spread = np.asarray(y, float) - (hr * np.asarray(x, float) + intercept)
        half_life = _ou_half_life_mle(spread)

        res = CointResult(
            ticker_x=tx,
            ticker_y=ty,
            eg_pvalue=eg_pval,
            johansen_trace_stat=joh_trace,
            johansen_critical_5=joh_cv5,
            hedge_ratio=hr,
            intercept=intercept,
            half_life_days=half_life,
            adf_pvalue=adf_pval,
            spread_mean=float(np.mean(spread)),
            spread_std=float(res_std),
        )
        res.score = score_pair(res)
        results.append(res)

    return sorted(results, key=lambda r: r.score, reverse=True)
