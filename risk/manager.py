"""
Risk management layer.

Implements portfolio-level controls:
  - Position limits (max pairs open simultaneously)
  - Gross/net exposure caps
  - Maximum drawdown circuit breaker
  - Pair-level correlation risk (avoid correlated pairs blowing up together)
  - Daily VaR monitor (parametric Gaussian VaR)
  - Kelly criterion position sizing (optional overlay)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_open_pairs:     int   = 5        # max simultaneous pair positions
    max_gross_exposure: float = 5.0      # as multiple of capital
    max_net_exposure:   float = 0.2      # long-short imbalance cap
    max_drawdown_pct:   float = 0.15     # 15% drawdown → halt trading
    var_confidence:     float = 0.99     # 99% 1-day VaR
    var_limit_pct:      float = 0.02     # 2% of NAV
    kelly_fraction:     float = 0.25     # fractional Kelly


class RiskManager:
    """
    Stateful risk checker — called before every trade entry.
    """

    def __init__(self, initial_capital: float, cfg: RiskConfig | None = None):
        self.capital = initial_capital
        self.cfg = cfg or RiskConfig()
        self._peak_nav = initial_capital
        self._daily_returns: list[float] = []
        self._halted = False

    def update_nav(self, nav: float) -> None:
        if nav > self._peak_nav:
            self._peak_nav = nav
        if len(self._daily_returns) > 0:
            prev = self.capital
            self._daily_returns.append((nav - prev) / prev)
        self.capital = nav

        # Circuit breaker
        drawdown = (self._peak_nav - nav) / self._peak_nav
        if drawdown >= self.cfg.max_drawdown_pct:
            self._halted = True

    def can_open_trade(
        self,
        num_open: int,
        gross_exposure: float,
        net_exposure: float,
    ) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if self._halted:
            return False, "HALTED: max drawdown exceeded"
        if num_open >= self.cfg.max_open_pairs:
            return False, f"REJECTED: max open pairs ({self.cfg.max_open_pairs})"
        if gross_exposure / max(self.capital, 1) > self.cfg.max_gross_exposure:
            return False, "REJECTED: gross exposure limit"
        if abs(net_exposure) / max(self.capital, 1) > self.cfg.max_net_exposure:
            return False, "REJECTED: net exposure limit"
        return True, "OK"

    def parametric_var(self, nav: float) -> float:
        """1-day parametric VaR (Gaussian, 99% confidence)."""
        if len(self._daily_returns) < 20:
            return 0.0
        mu = np.mean(self._daily_returns[-60:])
        sigma = np.std(self._daily_returns[-60:])
        # VaR = -(mu - z * sigma) * nav
        z = 2.326  # 99th percentile
        return max(0.0, -(mu - z * sigma) * nav)

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Fractional Kelly criterion for position sizing.
        f* = (p * b - q) / b, where b = avg_win/avg_loss
        """
        if avg_loss <= 0 or avg_win <= 0:
            return self.cfg.kelly_fraction
        b = avg_win / avg_loss
        q = 1 - win_rate
        full_kelly = (win_rate * b - q) / b
        return max(0.0, min(full_kelly * self.cfg.kelly_fraction, 1.0))

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def current_drawdown(self) -> float:
        if self._peak_nav == 0:
            return 0.0
        return (self._peak_nav - self.capital) / self._peak_nav


# ── Correlation risk analysis ──────────────────────────────────────────────────

def pair_correlation_matrix(spread_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compute pairwise correlations between pair spreads.
    High correlation → concentration risk.
    """
    df = pd.DataFrame(spread_dict).dropna()
    return df.corr()


def concentration_herfindahl(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index of position concentration.
    HHI = Σ wi² → 1/N (perfectly diversified) to 1 (fully concentrated)
    """
    w = np.abs(weights)
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    return float(np.sum(w**2))


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(returns: np.ndarray, freq: int = 252) -> dict:
    """
    Compute standard quant performance metrics from a daily return series.
    """
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return {}

    total_ret  = np.prod(1 + r) - 1
    ann_ret    = (1 + total_ret) ** (freq / len(r)) - 1
    ann_vol    = np.std(r, ddof=1) * np.sqrt(freq)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino — downside deviation only
    downside   = r[r < 0]
    dd_vol     = np.std(downside, ddof=1) * np.sqrt(freq) if len(downside) > 1 else ann_vol
    sortino    = ann_ret / dd_vol if dd_vol > 0 else 0.0

    # Calmar
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    dd_series   = (cum - running_max) / running_max
    max_dd      = float(np.min(dd_series))
    calmar      = -ann_ret / max_dd if max_dd < 0 else 0.0

    # Win rate
    win_rate    = float(np.mean(r > 0))

    # Skew & Kurtosis (tail risk indicators)
    from scipy import stats as sps
    skew  = float(sps.skew(r))
    kurt  = float(sps.kurtosis(r))

    # VaR & CVaR at 95%
    var_95  = float(np.percentile(r, 5))
    cvar_95 = float(np.mean(r[r <= var_95]))

    return {
        "total_return_pct":  total_ret * 100,
        "annual_return_pct": ann_ret * 100,
        "annual_vol_pct":    ann_vol * 100,
        "sharpe_ratio":      sharpe,
        "sortino_ratio":     sortino,
        "calmar_ratio":      calmar,
        "max_drawdown_pct":  max_dd * 100,
        "win_rate_pct":      win_rate * 100,
        "skewness":          skew,
        "excess_kurtosis":   kurt,
        "var_95_pct":        var_95 * 100,
        "cvar_95_pct":       cvar_95 * 100,
        "num_observations":  len(r),
    }
