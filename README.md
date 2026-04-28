# 🛡️ Digital Shield 2 — ML-Powered Railway Ticket Anomaly Detection

> **Version 3** · Isolation Forest + LOF + Weighted Hybrid Boost · Risk Scale 0–10

Digital Shield 2 is an end-to-end machine learning pipeline for detecting suspicious and anomalous booking patterns in railway ticket data. It combines unsupervised ML models with a rule-based boosting layer to produce human-readable risk scores, and exposes results through an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features-engineered)
- [Risk Scoring](#-risk-scoring)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Pipeline](#-running-the-pipeline)
- [Dashboard](#-dashboard)
- [Output Files](#-output-files)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

Digital Shield 2 connects to a PostgreSQL database (Supabase), loads railway ticket records, engineers 14 risk-relevant features, trains an **Isolation Forest** model cross-validated with **Local Outlier Factor (LOF)**, applies a **weighted multiplicative boost** using rule-based signals, and generates a risk score between **0 and 10** for every record.

Records scoring above **6.0** are classified as **High Risk** and saved to dedicated output files for investigation.

### Key Capabilities

| Capability | Detail |
|---|---|
| 🤖 ML Engine | Isolation Forest (n=150) + optional LOF cross-validation |
| 📊 Risk Scale | 0–10 (higher = more suspicious) |
| ⚡ Performance | Fully vectorised — no Python loops in feature engineering |
| 💬 Explainability | 1–3 human-readable reasons per flagged record |
| 📈 Dashboard | Interactive Streamlit + Plotly visualisation |
| 🗃️ Output | CSV + JSON reports for high-risk and all-score records |

---

## 🏗️ Architecture

```
PostgreSQL (Supabase)
        │
        ▼
  data_loader.py        ← STEP 1: Selective column load, dtype optimisation
        │
        ▼
   cleaner.py           ← STEP 2: String sanitisation, datetime parsing, age clipping
        │
        ▼
  features.py           ← STEP 3: 14 features across 8 categories (vectorised)
        │
        ▼
    model.py            ← STEP 4–5: RobustScaler → Isolation Forest → MinMax [0,10]
        │
        ▼
   scoring.py           ← STEP 6: Weighted multiplicative hybrid boosting, cap at 10
        │
        ▼
  explainer.py          ← STEP 7: Vectorised reason generation (up to 3 per record)
        │
        ▼
risk_detection_engine   ← STEP 8: Extract high-risk (>6.0), save CSV/JSON
        │
        ▼
  dashboard.py          ← Interactive Streamlit dashboard (reads outputs/)
```

---

## 🧠 Features Engineered

14 features across 8 categories, all computed using vectorised Pandas — zero Python loops:

| # | Category | Features |
|---|---|---|
| 1 | **Age** | `is_minor` |
| 2 | **Time** | `lead_time_hours`, `is_last_minute`, `booking_hour`, `is_nighttime_booking` |
| 3 | **Group (per PNR)** | `group_size`, `is_bulk_booking`, `minors_per_pnr`, `adults_per_pnr`, `normalized_group_size` |
| 4 | **User Patterns** | `bookings_per_user`, `unique_passengers_per_user`, `user_behavioral_variance` |
| 5 | **Network Anomaly** | `ip_booking_count`, `ip_anomaly_score`, `bank_usage_count` |
| 6 | **Route Intelligence** | `route_frequency`, `route_rarity_score`, `unusual_route_flag` |
| 7 | **Behavioral Signals** | `high_variance_user`, `repeated_group_flag`, `user_route_count` |
| 8 | **Normalised** | `normalized_group_size` |

> Features fed to the model: `age`, `lead_time_hours`, `group_size`, `bookings_per_user`, `unique_passengers_per_user`, `ip_booking_count`, `route_frequency`, `bank_usage_count`, `user_route_count`, `route_rarity_score`, `ip_anomaly_score`, `user_behavioral_variance`

---

## 🎯 Risk Scoring

### Step 1 — ML Base Score (Isolation Forest)
```
raw_score  = IsolationForest.decision_function(X)   # lower = more anomalous
inverted   = -raw_score
risk_score = MinMaxScaler(0, 10).fit_transform(inverted)
```

### Step 2 — LOF Cross-Validation
Local Outlier Factor flags additional anomalies. Records flagged by both models receive a +10% boost.

### Step 3 — Weighted Multiplicative Boost
```
final_score = base_ml_score × (1.0 + Σ triggered_weights)
final_score = clip(final_score, max=10.0)
```

| Signal | Boost Weight | Condition |
|---|---|---|
| Minor in bulk booking | +0.40 | Age < 18 AND group ≥ 3 per PNR |
| Last-minute group | +0.30 | Booked < 24h before journey AND group ≥ 3 |
| High IP anomaly | +0.20 | IP booking count ≥ 95th percentile |
| High user variance | +0.20 | Unique passengers ≥ 90th percentile |
| Unusual route | +0.15 | Route rarity score ≥ 0.90 |
| Nighttime group booking | +0.15 | Booked 11PM–5AM AND group ≥ 3 |
| LOF cross-validation | +0.10 | Also flagged by Local Outlier Factor |

### Risk Categories

| Score Range | Category | Label |
|---|---|---|
| 9.0 – 10.0 | 🔴 CRITICAL | Immediate investigation required |
| 7.0 – 8.9 | 🟠 HIGH | High priority review |
| 5.0 – 6.9 | 🟡 MEDIUM | Monitor closely |
| 0.0 – 4.9 | 🟢 LOW | Normal activity |

---

## 📁 Project Structure

```
DigitalShield2/
├── risk_detection_engine.py    # Main pipeline orchestrator (run this)
├── dashboard.py                # Streamlit interactive dashboard
├── probe_schema.py             # Utility: probe DB schema and sample data
│
├── src/
│   ├── config.py               # All constants, thresholds, hyperparameters
│   ├── data_loader.py          # PostgreSQL data loading + dtype optimisation
│   ├── cleaner.py              # Data cleaning and datetime parsing
│   ├── features.py             # Feature engineering (14 features, vectorised)
│   ├── model.py                # Isolation Forest + LOF, MinMax [0–10] calibration
│   ├── scoring.py              # Weighted multiplicative hybrid risk scoring
│   ├── explainer.py            # Human-readable reason generation
│   ├── logger.py               # Logging setup
│   └── __init__.py
│
├── outputs/                    # Generated output files (gitignored)
│   ├── all_risk_scores.csv     # All records with risk scores
│   ├── high_risk_records.csv   # Records with risk_score > 6.0
│   └── high_risk_records.json  # Same, in JSON format
│
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Access to the Supabase PostgreSQL database

### 1. Clone the Repository
```bash
git clone https://github.com/Gyan0205/digitalshield.git
cd digitalshield
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn sqlalchemy psycopg2-binary streamlit plotly
```

Or if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

All settings are centralised in `src/config.py`:

```python
# Risk threshold — records above this score are "High Risk"
HIGH_RISK_THRESHOLD = 6.0

# Isolation Forest hyperparameters
ISOLATION_FOREST = {
    "n_estimators": 150,
    "contamination": 0.03,
    "max_features": 0.8,
    "random_state": 42,
}

# LOF cross-validation
LOF_ENABLED = True
LOF_PARAMS = {"n_neighbors": 20, "contamination": 0.03}

# Boost signal thresholds
HIGH_IP_PERCENTILE = 95
HIGH_USER_VARIANCE_PERCENTILE = 90
UNUSUAL_ROUTE_RARITY_THRESHOLD = 0.90
```

---

## 🚀 Running the Pipeline

### Run the Full ML Detection Pipeline
```bash
python risk_detection_engine.py
```

This will:
1. Load data from the PostgreSQL database
2. Clean and engineer features
3. Train Isolation Forest + LOF
4. Apply hybrid risk scoring (0–10 scale)
5. Generate explainability reasons
6. Save results to `outputs/`

**Expected output:**
```
================================================================
  DIGITAL SHIELD 2 v3 — ML Risk Detection Engine
  Isolation Forest + LOF + Weighted Hybrid Boosting
================================================================
STEP 1: Loading data from database
  Loaded 523,412 rows x 16 cols in 4.21s
STEP 2: Cleaning data
STEP 3: Engineering features (14 features, 8 categories)
STEP 4: Preparing features for model
STEP 5: Training Isolation Forest
STEP 6: Applying weighted hybrid risk scoring
STEP 7: Generating explainability reasons
STEP 8: Extracting high-risk records (threshold > 6.0)
================================================================
  PIPELINE COMPLETE
  Total records processed : 523,412
  High-risk records       : 15,702
  Avg score               : 3.41
  95th percentile         : 7.83
================================================================
```

### Probe DB Schema (Optional)
```bash
python probe_schema.py
```

---

## 📊 Dashboard

After running the pipeline, launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

### Dashboard Features

| Section | Description |
|---|---|
| **KPI Cards** | Total records, high-risk count, critical (>9.0), minors at risk, avg score, filtered view |
| **Risk Distribution** | Histogram of all risk scores with threshold marker |
| **Risk Categories Pie** | Critical / High / Medium / Low breakdown |
| **Age vs Risk Scatter** | 5,000-record sample coloured by score |
| **Top High-Risk Routes** | Bar chart of top 10 routes by alert volume |
| **Gender Risk Analysis** | Average risk score by gender |
| **Top Risk Reasons** | Most common anomaly reasons across flagged records |
| **Top Suspicious Users** | Top 15 users by alert count, coloured by avg risk |
| **Filtered Data Table** | Searchable / filterable table with progress bars |

### Sidebar Filters
- **Risk Score Range** — slider from 0.0 to 10.0 (default: 6.0–10.0)
- **Age Range** — slider across full age range
- **Gender** — dropdown filter
- **Search** — free-text search by PNR, name, or user ID

---

## 📄 Output Files

All outputs are saved to the `outputs/` directory:

| File | Description |
|---|---|
| `outputs/all_risk_scores.csv` | All records with `risk_score` + `reason` columns |
| `outputs/high_risk_records.csv` | Only records with `risk_score > 6.0`, sorted descending |
| `outputs/high_risk_records.json` | Same high-risk records in JSON format |

### Output Columns

```
pnrno | user_id | psgn_name | age | sex | from_stn | to_stn | risk_score | reason
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | Isolation Forest, LOF, RobustScaler, MinMaxScaler |
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Vectorised numerical operations |
| `sqlalchemy` + `psycopg2` | PostgreSQL database connection |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts and visualisations |

---

## 📌 Notes

- **Large output files** (`*.csv`, `*.json`, `outputs/`) are excluded from git via `.gitignore`. Run the pipeline locally to regenerate them.
- **Database credentials** are stored in `src/config.py`. For production, move these to environment variables or Streamlit secrets.
- The pipeline is designed for **large datasets** (500K+ records). RobustScaler is used instead of StandardScaler for better outlier resistance.

---

*Digital Shield 2 v3 — Built for railway ticket fraud and anomaly detection.*