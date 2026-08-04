"""
Signal generation engine.

Two spread estimators:
  1. Static OLS hedge ratio (baseline)
  2. Kalman Filter dynamic hedge ratio (production-quality)

Signal logic follows a standard z-score mean-reversion framework
with hysteresis (different entry/exit thresholds) to avoid overtrading.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.models import PairConfig, SpreadSnapshot, SignalState


# ─── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanHedgeFilter:
    """
    Online Kalman filter to estimate a time-varying hedge ratio.

    State vector: [hedge_ratio, intercept]
    Observation:  y_t = x_t * hedge_ratio_t + intercept_t + noise

    This is a standard linear Gaussian state-space model (local level).
    The filter runs in O(1) per timestep — ideal for real-time use.
    """

    def __init__(self, delta: float = 1e-4, ve: float = 0.001):
        """
        Args:
            delta: Process noise variance (controls how fast β can change).
                   Smaller → smoother / slower adaptation.
            ve:    Observation noise variance.
        """
        # State estimate [β, α]
        self.theta = np.zeros(2)
        # State covariance
        self.P = np.eye(2) * 1.0
        # Process noise (random walk on β)
        self.Q = delta / (1 - delta) * np.eye(2)
        self.ve = ve  # measurement noise
        self._initialised = False

    def update(self, x_t: float, y_t: float) -> tuple[float, float, float]:
        """
        One-step Kalman update.

        Returns: (hedge_ratio, intercept, spread)
        """
        F = np.array([x_t, 1.0])  # observation matrix

        if not self._initialised:
            # Warm-start the state on first observation
            self.theta = np.array([1.0, 0.0])
            self._initialised = True

        # Predict
        P_pred = self.P + self.Q

        # Innovation
        y_hat   = F @ self.theta
        innov   = y_t - y_hat
        S       = F @ P_pred @ F + self.ve  # innovation covariance
        K       = P_pred @ F / S            # Kalman gain

        # Update
        self.theta = self.theta + K * innov
        self.P     = (np.eye(2) - np.outer(K, F)) @ P_pred

        hedge = self.theta[0]
        inter = self.theta[1]
        spread = y_t - hedge * x_t - inter
        return hedge, inter, spread


# ─── Spread Computer ───────────────────────────────────────────────────────────

def compute_spread_series(
    price_x: pd.Series,
    price_y: pd.Series,
    hedge_ratio: float,
    intercept: float,
    use_kalman: bool = True,
    delta: float = 1e-4,
) -> pd.DataFrame:
    """
    Compute the spread series between y and hedge_ratio*x + intercept.

    If use_kalman=True, the hedge ratio updates dynamically at each step,
    which better handles non-stationarity in the cointegration relationship.

    Returns DataFrame with columns: timestamp, spread, hedge_ratio
    """
    xs = np.asarray(price_x.values, float)
    ys = np.asarray(price_y.values, float)
    n  = len(xs)

    spreads  = np.empty(n)
    hrs      = np.empty(n)
    intercepts = np.empty(n)

    if use_kalman:
        kf = KalmanHedgeFilter(delta=delta)
        for i in range(n):
            hr, ic, sp = kf.update(xs[i], ys[i])
            spreads[i]    = sp
            hrs[i]        = hr
            intercepts[i] = ic
    else:
        spreads    = ys - (hedge_ratio * xs + intercept)
        hrs        = np.full(n, hedge_ratio)
        intercepts = np.full(n, intercept)

    return pd.DataFrame({
        "timestamp":   price_x.index if hasattr(price_x, "index") else range(n),
        "spread":      spreads,
        "hedge_ratio": hrs,
        "intercept":   intercepts,
    })


# ─── Z-Score & Signal Generation ──────────────────────────────────────────────

def compute_zscore(spread: pd.Series, lookback: int = 60) -> pd.DataFrame:
    """
    Rolling z-score normalisation.
    Uses an expanding window until we have `lookback` observations,
    then switches to a rolling window.
    """
    roll_mean = spread.rolling(window=lookback, min_periods=lookback//2).mean()
    roll_std  = spread.rolling(window=lookback, min_periods=lookback//2).std()

    z = (spread - roll_mean) / roll_std.replace(0, np.nan)

    return pd.DataFrame({
        "spread":      spread,
        "spread_mean": roll_mean,
        "spread_std":  roll_std,
        "z_score":     z,
    })


def generate_signals(
    pair_cfg: PairConfig,
    price_x: pd.Series,
    price_y: pd.Series,
    use_kalman: bool = True,
) -> list[SpreadSnapshot]:
    """
    Full signal generation pipeline for one pair.

    Returns a list of SpreadSnapshot objects, one per timestep.
    The signal field implements hysteretic entry/exit logic:

      - FLAT  → |z| < entry_z:  stay flat
      - ENTER → |z| >= entry_z: enter position
      - HOLD  → position open, |z| > exit_z: hold
      - EXIT  → position open, |z| <= exit_z or |z| >= stop_z: exit
    """
    # 1. Spread computation
    spread_df = compute_spread_series(
        price_x, price_y,
        hedge_ratio=pair_cfg.hedge_ratio,
        intercept=pair_cfg.intercept,
        use_kalman=use_kalman,
    )

    # 2. Z-score
    z_df = compute_zscore(spread_df["spread"], pair_cfg.lookback)

    # 3. Signal state machine
    snapshots: list[SpreadSnapshot] = []
    state = SignalState.FLAT

    dates = price_x.index if hasattr(price_x, "index") else range(len(price_x))

    for i, ts in enumerate(dates):
        z = z_df["z_score"].iloc[i]
        if pd.isna(z):
            snapshots.append(SpreadSnapshot(
                timestamp=ts,
                spread=float(z_df["spread"].iloc[i]),
                z_score=0.0,
                spread_mean=float(z_df["spread_mean"].iloc[i]) if not pd.isna(z_df["spread_mean"].iloc[i]) else 0.0,
                spread_std=float(z_df["spread_std"].iloc[i]) if not pd.isna(z_df["spread_std"].iloc[i]) else 1.0,
                signal=SignalState.FLAT,
            ))
            continue

        az = abs(z)

        # State transitions
        if state == SignalState.FLAT:
            if az >= pair_cfg.entry_z:
                state = SignalState.ENTER
            else:
                state = SignalState.FLAT
        elif state == SignalState.ENTER:
            state = SignalState.HOLD
        elif state == SignalState.HOLD:
            if az <= pair_cfg.exit_z or az >= pair_cfg.stop_z:
                state = SignalState.EXIT
            else:
                state = SignalState.HOLD
        elif state == SignalState.EXIT:
            state = SignalState.FLAT

        snapshots.append(SpreadSnapshot(
            timestamp=ts,
            spread=float(z_df["spread"].iloc[i]),
            z_score=float(z),
            spread_mean=float(z_df["spread_mean"].iloc[i]) if not pd.isna(z_df["spread_mean"].iloc[i]) else 0.0,
            spread_std=float(z_df["spread_std"].iloc[i]) if not pd.isna(z_df["spread_std"].iloc[i]) else 1.0,
            signal=state,
        ))

    return snapshots
