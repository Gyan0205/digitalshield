"""
explainer.py — Dynamic, data-driven explainability engine
Generates 1-3 human-readable risk reasons per record.
Fully vectorised using np.where + vectorised string join.
"""

import time
import numpy as np
import pandas as pd

from src.config import (
    HIGH_IP_PERCENTILE,
    HIGH_USER_VARIANCE_PERCENTILE,
    UNUSUAL_ROUTE_RARITY_THRESHOLD,
)
from src.logger import get_logger

log = get_logger()

# Reason templates ordered by severity (most severe first)
# Each entry: (condition_builder, template_string, priority)
REASON_REGISTRY = [
    {
        "id": "minor_group",
        "priority": 1,
        "template": "Minor (age {age}) detected in group of {group_size}",
    },
    {
        "id": "last_minute",
        "priority": 2,
        "template": "Last-minute booking ({lead_time}h before journey)",
    },
    {
        "id": "high_ip",
        "priority": 3,
        "template": "High IP usage ({ip_count} bookings from same IP)",
    },
    {
        "id": "multi_pax",
        "priority": 4,
        "template": "Multiple passengers ({unique_pax}) booked under one user",
    },
    {
        "id": "unusual_route",
        "priority": 5,
        "template": "Unusual travel route ({route}, rarity: {rarity:.0%})",
    },
    {
        "id": "bulk_booking",
        "priority": 6,
        "template": "Bulk booking ({group_size} passengers per PNR)",
    },
    {
        "id": "nighttime",
        "priority": 7,
        "template": "Nighttime booking (booked at {hour}:00)",
    },
    {
        "id": "repeated_group",
        "priority": 8,
        "template": "Repeated group pattern (same user + route, {count}x)",
    },
]


def generate_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 1-3 human-readable risk reasons for each record.

    Uses vectorised np.where to build reason arrays, then joins
    the top 3 most relevant reasons per record.

    Args:
        df: DataFrame with all feature columns and risk_score.

    Returns:
        DataFrame with 'reason' column added.
    """
    t0 = time.perf_counter()
    log.info("STEP 7: Generating explainability reasons")

    n = len(df)

    # Pre-compute thresholds
    ip_high = df["ip_booking_count"].quantile(HIGH_IP_PERCENTILE / 100) if "ip_booking_count" in df.columns else 999999
    uvar_high = df["unique_passengers_per_user"].quantile(HIGH_USER_VARIANCE_PERCENTILE / 100) if "unique_passengers_per_user" in df.columns else 999999

    # ── Build reason arrays (vectorised) ─────────────────────────────────
    # Each is an array of strings: non-empty where condition is True

    reasons = []

    # 1. Minor in group
    if "is_minor" in df.columns and "group_size" in df.columns:
        r = np.where(
            (df["is_minor"] == 1) & (df["group_size"] >= 2),
            "Minor (age " + df["age"].astype(str) + ") in group of " + df["group_size"].astype(str),
            "",
        )
        reasons.append(r)

    # 2. Last-minute booking
    if "is_last_minute" in df.columns:
        lead_str = df["lead_time_hours"].round(0).astype(int).astype(str)
        r = np.where(
            df["is_last_minute"] == 1,
            "Last-minute booking (" + lead_str + "h before journey)",
            "",
        )
        reasons.append(r)

    # 3. High IP usage
    if "ip_booking_count" in df.columns:
        ip_high_int = int(ip_high)
        r = np.where(
            df["ip_booking_count"] >= ip_high,
            "High IP usage (" + df["ip_booking_count"].astype(str) + " bookings, threshold: " + str(ip_high_int) + ")",
            "",
        )
        reasons.append(r)

    # 4. Multiple passengers per user
    if "unique_passengers_per_user" in df.columns:
        r = np.where(
            df["unique_passengers_per_user"] >= uvar_high,
            "Multiple passengers (" + df["unique_passengers_per_user"].astype(str) + ") under one user",
            "",
        )
        reasons.append(r)

    # 5. Unusual route
    if "unusual_route_flag" in df.columns and "route_rarity_score" in df.columns:
        rarity_pct = (df["route_rarity_score"] * 100).round(0).astype(int).astype(str)
        r = np.where(
            df["unusual_route_flag"] == 1,
            "Unusual route (" + df["from_stn"].astype(str) + "->" + df["to_stn"].astype(str) + ", rarity " + rarity_pct + "%)",
            "",
        )
        reasons.append(r)

    # 6. Bulk booking
    if "is_bulk_booking" in df.columns:
        r = np.where(
            df["is_bulk_booking"] == 1,
            "Bulk booking (" + df["group_size"].astype(str) + " passengers per PNR)",
            "",
        )
        reasons.append(r)

    # 7. Nighttime booking
    if "is_nighttime_booking" in df.columns:
        r = np.where(
            df["is_nighttime_booking"] == 1,
            "Nighttime booking (at " + df["booking_hour"].astype(str) + ":00)",
            "",
        )
        reasons.append(r)

    # 8. Repeated group pattern
    if "repeated_group_flag" in df.columns and "user_route_count" in df.columns:
        r = np.where(
            df["repeated_group_flag"] == 1,
            "Repeated group pattern (same user+route, " + df["user_route_count"].astype(str) + "x)",
            "",
        )
        reasons.append(r)

    # ── Combine reasons (max 3 per record) ───────────────────────────────
    if reasons:
        all_reasons = np.column_stack(reasons)
        df["reason"] = _join_top_reasons(all_reasons, max_reasons=3)
    else:
        df["reason"] = "Anomalous statistical pattern"

    elapsed = time.perf_counter() - t0
    log.info(f"  Reasons generated ({elapsed:.2f}s)")

    return df


def _join_top_reasons(reason_matrix: np.ndarray, max_reasons: int = 3) -> list[str]:
    """
    Join non-empty reasons per row, keeping at most max_reasons.
    Optimised: iterates through numpy rows (faster than df.apply).
    """
    result = []
    for row in reason_matrix:
        filtered = [r for r in row if r]
        if filtered:
            result.append("; ".join(filtered[:max_reasons]))
        else:
            result.append("Anomalous statistical pattern")
    return result
