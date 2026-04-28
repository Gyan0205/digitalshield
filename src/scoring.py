"""
scoring.py — Weighted multiplicative hybrid risk scoring
Prevents score inflation seen in flat additive approaches.
"""

import time
import numpy as np
import pandas as pd

from src.config import (
    BOOST_WEIGHTS,
    HIGH_IP_PERCENTILE,
    HIGH_USER_VARIANCE_PERCENTILE,
    UNUSUAL_ROUTE_RARITY_THRESHOLD,
)
from src.logger import get_logger

log = get_logger()


def apply_hybrid_scoring(
    df: pd.DataFrame,
    base_scores: np.ndarray,
    lof_flags: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Apply weighted multiplicative boost to ML base scores.

    Formula: risk_score = base_ml_score * (1.0 + sum_of_weighted_signals)
    Capped at 10.0.

    This approach ensures:
      - Low ML scores stay low even if rules fire (no artificial inflation)
      - High ML scores get amplified when rules also confirm suspicion
      - Records must be BOTH statistically anomalous AND match patterns

    Args:
        df: DataFrame with all feature columns.
        base_scores: Calibrated 0-10 ML risk scores.
        lof_flags: Optional LOF anomaly flags (True = anomaly).

    Returns:
        DataFrame with 'risk_score' column added.
    """
    t0 = time.perf_counter()
    log.info("STEP 6: Applying weighted hybrid risk scoring")

    df["risk_score"] = base_scores

    # Pre-compute thresholds once
    ip_high = df["ip_anomaly_score"].quantile(HIGH_IP_PERCENTILE / 100) if "ip_anomaly_score" in df.columns else 1.0
    uvar_high = df["user_behavioral_variance"].quantile(HIGH_USER_VARIANCE_PERCENTILE / 100) if "user_behavioral_variance" in df.columns else 1.0

    # ── Build boost factor vectorised ────────────────────────────────────
    boost = pd.Series(0.0, index=df.index, dtype=np.float64)

    # Minor in bulk booking
    if "is_minor" in df.columns and "is_bulk_booking" in df.columns:
        boost += BOOST_WEIGHTS["minor_in_bulk"] * (
            (df["is_minor"] == 1) & (df["is_bulk_booking"] == 1)
        ).astype(float)

    # Last-minute + large group
    if "is_last_minute" in df.columns and "group_size" in df.columns:
        boost += BOOST_WEIGHTS["last_minute_group"] * (
            (df["is_last_minute"] == 1) & (df["group_size"] >= 3)
        ).astype(float)

    # High IP anomaly score
    if "ip_anomaly_score" in df.columns:
        boost += BOOST_WEIGHTS["high_ip_anomaly"] * (
            df["ip_anomaly_score"] >= ip_high
        ).astype(float)

    # High user behavioral variance
    if "user_behavioral_variance" in df.columns:
        boost += BOOST_WEIGHTS["high_user_variance"] * (
            df["user_behavioral_variance"] >= uvar_high
        ).astype(float)

    # Unusual route
    if "route_rarity_score" in df.columns:
        boost += BOOST_WEIGHTS["unusual_route"] * (
            df["route_rarity_score"] >= UNUSUAL_ROUTE_RARITY_THRESHOLD
        ).astype(float)

    # Nighttime booking + group >= 3
    if "is_nighttime_booking" in df.columns and "group_size" in df.columns:
        boost += BOOST_WEIGHTS["nighttime_group"] * (
            (df["is_nighttime_booking"] == 1) & (df["group_size"] >= 3)
        ).astype(float)

    # LOF cross-validation boost
    if lof_flags is not None:
        boost += BOOST_WEIGHTS["lof_cross_validated"] * pd.Series(
            lof_flags.astype(float), index=df.index
        )

    # ── Apply multiplicative boost ───────────────────────────────────────
    multiplier = 1.0 + boost
    df["risk_score"] = (df["risk_score"] * multiplier).clip(upper=10.0)

    boosted_count = (boost > 0).sum()
    elapsed = time.perf_counter() - t0

    log.info(f"  Boosted {boosted_count:,} records (multiplicative)")
    log.info(
        f"  Final scores: min={df['risk_score'].min():.4f}, "
        f"mean={df['risk_score'].mean():.4f}, max={df['risk_score'].max():.4f}"
    )
    log.info(f"  Scoring complete ({elapsed:.2f}s)")

    return df
