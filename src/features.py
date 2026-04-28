"""
features.py — Advanced feature engineering (fully vectorised)
14 features across 8 categories. Zero Python loops.
"""

import time
import pandas as pd
import numpy as np

from src.logger import get_logger

log = get_logger()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all risk-relevant features using only vectorised Pandas operations.

    Feature categories:
      1. Age features (is_minor)
      2. Time features (lead_time_hours, is_last_minute, booking_hour, is_nighttime)
      3. Group features (group_size, is_bulk_booking, minors_per_pnr, adults_per_pnr)
      4. User pattern features (bookings_per_user, unique_passengers_per_user, user_behavioral_variance)
      5. Network anomaly features (ip_booking_count, ip_anomaly_score, bank_usage_count)
      6. Route intelligence (route_frequency, route_rarity_score, unusual_route_flag)
      7. Behavioral signals (high_variance_user, repeated_group_flag)
      8. Normalised features (normalized_group_size)

    Returns:
        DataFrame with all feature columns added (mutates in-place for efficiency).
    """
    t0 = time.perf_counter()
    log.info("STEP 3: Engineering features (14 features, 8 categories)")

    # ══════════════════════════════════════════════════════════════════════
    # 1. AGE FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["is_minor"] = (df["age"] < 18).astype(np.int8)
    log.debug("  [1/8] Age features")

    # ══════════════════════════════════════════════════════════════════════
    # 2. TIME FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["lead_time_hours"] = (
        (df["jrny_date_dt"] - df["booking_datetime"]).dt.total_seconds() / 3600
    )
    # Fill NaT-caused NaNs with median, clamp negatives to 0
    median_lead = df["lead_time_hours"].median()
    if pd.isna(median_lead):
        median_lead = 48.0
    df["lead_time_hours"] = df["lead_time_hours"].fillna(median_lead).clip(lower=0)
    df["is_last_minute"] = (df["lead_time_hours"] < 24).astype(np.int8)

    # Booking hour + nighttime flag (11PM-5AM)
    df["booking_hour"] = df["booking_datetime"].dt.hour.fillna(12).astype(np.int8)
    df["is_nighttime_booking"] = (
        (df["booking_hour"] >= 23) | (df["booking_hour"] <= 5)
    ).astype(np.int8)
    log.debug("  [2/8] Time features (lead_time, last_minute, booking_hour, nighttime)")

    # ══════════════════════════════════════════════════════════════════════
    # 3. GROUP FEATURES (per PNR)
    # ══════════════════════════════════════════════════════════════════════
    df["group_size"] = df.groupby("pnrno")["pnrno"].transform("count")
    df["is_bulk_booking"] = (df["group_size"] >= 3).astype(np.int8)

    # Minors and adults per PNR
    df["_is_minor_int"] = df["is_minor"].astype(int)
    df["minors_per_pnr"] = df.groupby("pnrno")["_is_minor_int"].transform("sum")
    df["adults_per_pnr"] = df["group_size"] - df["minors_per_pnr"]

    # Normalised group size (0-1 scale)
    max_group = df["group_size"].max()
    df["normalized_group_size"] = (
        df["group_size"] / max_group if max_group > 0 else 0
    )
    df.drop(columns=["_is_minor_int"], inplace=True)
    log.debug("  [3/8] Group features (group_size, bulk, minors/adults per PNR)")

    # ══════════════════════════════════════════════════════════════════════
    # 4. USER PATTERN FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["bookings_per_user"] = df.groupby("user_id")["user_id"].transform("count")
    df["unique_passengers_per_user"] = df.groupby("user_id")["psgn_name"].transform("nunique")

    # User behavioral variance = unique_passengers / total_bookings
    # High variance = booking many different people = suspicious
    df["user_behavioral_variance"] = (
        df["unique_passengers_per_user"] / df["bookings_per_user"].clip(lower=1)
    ).astype(np.float32)
    log.debug("  [4/8] User pattern features (bookings, unique_pax, variance)")

    # ══════════════════════════════════════════════════════════════════════
    # 5. NETWORK ANOMALY FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["ip_booking_count"] = df.groupby("ip_addrs")["ip_addrs"].transform("count")
    df["bank_usage_count"] = df.groupby("bank_name")["bank_name"].transform("count")

    # IP anomaly score: percentile rank (0-1, higher = more anomalous)
    df["ip_anomaly_score"] = df["ip_booking_count"].rank(pct=True).astype(np.float32)
    log.debug("  [5/8] Network anomaly features (IP, bank, ip_anomaly_score)")

    # ══════════════════════════════════════════════════════════════════════
    # 6. ROUTE INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════
    df["route"] = df["from_stn"].astype(str) + "_" + df["to_stn"].astype(str)
    df["route_frequency"] = df.groupby("route")["route"].transform("count")

    # Route rarity score: inverse percentile (higher = rarer route)
    df["route_rarity_score"] = (
        1.0 - df["route_frequency"].rank(pct=True)
    ).astype(np.float32)

    route_q10 = df["route_frequency"].quantile(0.10)
    df["unusual_route_flag"] = (df["route_frequency"] <= route_q10).astype(np.int8)
    log.debug("  [6/8] Route intelligence (frequency, rarity_score, unusual_flag)")

    # ══════════════════════════════════════════════════════════════════════
    # 7. BEHAVIORAL SIGNALS
    # ══════════════════════════════════════════════════════════════════════
    high_var_threshold = df["unique_passengers_per_user"].quantile(0.90)
    df["high_variance_user"] = (
        df["unique_passengers_per_user"] >= high_var_threshold
    ).astype(np.int8)

    # Repeated group: same user + same route count
    df["user_route_key"] = df["user_id"].astype(str) + "_" + df["route"]
    df["user_route_count"] = df.groupby("user_route_key")["user_route_key"].transform("count")
    df["repeated_group_flag"] = (df["user_route_count"] >= 3).astype(np.int8)
    log.debug("  [7/8] Behavioral signals (high_variance_user, repeated_group)")

    # ══════════════════════════════════════════════════════════════════════
    # 8. SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t0
    log.info(f"  Feature engineering complete: 14 features in {elapsed:.2f}s")

    return df
