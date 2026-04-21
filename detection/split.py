"""
detection/split.py
------------------
Loads the raw transaction CSV and creates temporally-ordered train / val / test
splits with NO data leakage.

Split boundaries (relative to simulation start date derived from min timestamp):
    Train : days  0 – 59   (inclusive)
    Val   : days 60 – 74   (inclusive)
    Test  : days 75 – 89   (inclusive)

Rules:
    - No shuffling.
    - Split strictly on day index, not row index.
    - Each split contains only transactions whose timestamp falls in the window.
"""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/raw/transactions.csv")

TRAIN_START_DAY = 0
TRAIN_END_DAY   = 59
VAL_START_DAY   = 60
VAL_END_DAY     = 74
TEST_START_DAY  = 75
TEST_END_DAY    = 89


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_split(
    data_path: str | Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV, parse timestamps, and return (train_df, val_df, test_df).
    """
    is_inference = os.environ.get("AML_INFERENCE", "0") == "1"
    if is_inference:
        data_path = os.environ.get("AML_DATA_PATH", data_path)

    df = _load(data_path, is_inference=is_inference)
    
    if is_inference:
        empty_df = df.iloc[:0].copy()
        return empty_df, empty_df, df

    df = _add_day_index(df)

    train_df = df[(df["day_index"] >= TRAIN_START_DAY) & (df["day_index"] <= TRAIN_END_DAY)].copy()
    val_df   = df[(df["day_index"] >= VAL_START_DAY)   & (df["day_index"] <= VAL_END_DAY)].copy()
    test_df  = df[(df["day_index"] >= TEST_START_DAY)  & (df["day_index"] <= TEST_END_DAY)].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(data_path: str | Path, is_inference: bool = False) -> pd.DataFrame:
    """Read CSV and parse the timestamp column."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path.resolve()}")

    df = pd.read_csv(path, dtype=str)

    # In inference mode, fill missing required columns with dummies
    expected_cols = {
        "transaction_id", "sender", "receiver", "amount",
        "timestamp", "sender_country", "receiver_country",
        "laundering_flag", "campaign_id",
    }
    
    if is_inference:
        for col in expected_cols:
            if col not in df.columns:
                if col == "amount":
                    df[col] = "0.0"
                elif col == "laundering_flag":
                    df[col] = "0"
                else:
                    df[col] = ""

    # Re-cast to requested types + parse dates
    if "amount" in df.columns:
        df["amount"] = df["amount"].astype(float)
    if "laundering_flag" in df.columns:
        df["laundering_flag"] = df["laundering_flag"].astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    required_cols = expected_cols
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df


def _add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute integer day index relative to minimum timestamp in dataset."""
    start_date = df["timestamp"].min().normalize()  # floor to midnight
    df = df.copy()
    df["day_index"] = (df["timestamp"] - start_date).dt.days
    return df
