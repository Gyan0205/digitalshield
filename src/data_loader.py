"""
data_loader.py — Efficient data loading from PostgreSQL
Selective column loading, dtype optimisation, optional chunking.
"""

import time
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DB_URL, TABLE_NAME, REQUIRED_COLUMNS
from src.logger import get_logger

log = get_logger()


def load_data() -> pd.DataFrame:
    """
    Load data from tickets_1 using selective columns and optimised dtypes.

    Returns:
        pd.DataFrame with only the required columns, memory-optimised.
    """
    t0 = time.perf_counter()
    log.info("STEP 1: Loading data from database")

    engine = create_engine(DB_URL)

    # Build selective column query
    cols_sql = ", ".join(REQUIRED_COLUMNS)
    query = text(f"SELECT {cols_sql} FROM {TABLE_NAME}")

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    except Exception as e:
        log.error(f"Database query failed: {e}")
        raise
    finally:
        engine.dispose()

    # --- Dtype optimisation: downcast where safe ---
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int16")
    if "train_number" in df.columns:
        df["train_number"] = pd.to_numeric(df["train_number"], errors="coerce").astype("Int32")
    if "pnrno" in df.columns:
        df["pnrno"] = pd.to_numeric(df["pnrno"], errors="coerce").astype("Int64")

    elapsed = time.perf_counter() - t0
    mem_mb = df.memory_usage(deep=True).sum() / 1e6

    log.info(f"  Loaded {len(df):,} rows x {len(df.columns)} cols in {elapsed:.2f}s")
    log.info(f"  Memory usage: {mem_mb:.1f} MB")

    return df
