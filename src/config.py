"""
config.py — Centralised configuration for Digital Shield 2
All constants, thresholds, model hyperparameters, and DB credentials.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

DB_URL = (
    "postgresql://postgres.trssafmvdbeagsdnivcl:"
    "digitalshield%4023070802"
    "@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
)

TABLE_NAME = "tickets_1"

# Only load the columns we actually need (skip coach_no_seat_no, txn_no, etc.)
REQUIRED_COLUMNS = [
    "user_id", "psgn_name", "train_number", "cls",
    "txn_date", "txn_time", "ip_addrs", "jrny_date",
    "pnrno", "from_stn", "to_stn", "age", "sex",
    "quota", "txntype", "bank_name",
]

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

ISOLATION_FOREST = {
    "n_estimators": 150,
    "contamination": 0.03,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

LOF_ENABLED = True
LOF_PARAMS = {
    "n_neighbors": 20,
    "contamination": 0.03,
    "novelty": False,
    "n_jobs": -1,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS (fed to the model)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "age",
    "lead_time_hours",
    "group_size",
    "bookings_per_user",
    "unique_passengers_per_user",
    "ip_booking_count",
    "route_frequency",
    "bank_usage_count",
    "user_route_count",
    "route_rarity_score",
    "ip_anomaly_score",
    "user_behavioral_variance",
]

# ═══════════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════════

HIGH_RISK_THRESHOLD = 6.0

# Weighted boost signals (multiplicative approach)
# Final: risk_score = base_ml_score * (1.0 + sum_of_triggered_weights)
BOOST_WEIGHTS = {
    "minor_in_bulk":          0.40,  # Minor + bulk booking (>=3 per PNR)
    "last_minute_group":      0.30,  # Last-minute + group >= 3
    "high_ip_anomaly":        0.20,  # IP anomaly score > 95th pctile
    "high_user_variance":     0.20,  # User behavioral variance > 90th pctile
    "unusual_route":          0.15,  # Route rarity score > 0.90
    "nighttime_group":        0.15,  # Nighttime booking + group >= 3
    "lof_cross_validated":    0.10,  # LOF also flagged the record
}

# Percentile thresholds
HIGH_IP_PERCENTILE = 95
HIGH_USER_VARIANCE_PERCENTILE = 90
UNUSUAL_ROUTE_RARITY_THRESHOLD = 0.90

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_COLS = [
    "pnrno", "user_id", "psgn_name", "age", "sex",
    "from_stn", "to_stn", "risk_score", "reason",
]

OUTPUT_DIR = "outputs"
