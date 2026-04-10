"""
Feature engineering script for the video game dataset.

Steps:
1) Load processed splits from data/processed (output of data_preprocessing.py).
2) Add derived features (budgets, value ratios, engagement metrics, etc.).
3) Save engineered splits to data/preprocessed/train.csv and data/preprocessed/test.csv.
4) Log all actions to log/feature_enginnering.txt and console.
"""

import logging
import os
import math
from typing import Tuple

import pandas as pd

from utils import load_params, ensure_local_deps

PARAMS = load_params()
FE_PARAMS = PARAMS["feature_engineering"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, FE_PARAMS["source_subdir"])
PREPROCESSED_DIR = os.path.join(DATA_DIR, FE_PARAMS["output_subdir"])
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", FE_PARAMS["log_dir"])
LOG_FILE = os.path.join(LOG_DIR, "feature_enginnering.txt")

TRAIN_PROCESSED = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PROCESSED = os.path.join(PROCESSED_DIR, "test.csv")
TRAIN_PREPROCESSED = os.path.join(PREPROCESSED_DIR, "train.csv")
TEST_PREPROCESSED = os.path.join(PREPROCESSED_DIR, "test.csv")


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format, handlers=handlers)


def ensure_directories() -> None:
    for path in (DATA_DIR, PROCESSED_DIR, PREPROCESSED_DIR, LOG_DIR):
        os.makedirs(path, exist_ok=True)
    logging.info("Ensured data directories exist.")


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(TRAIN_PROCESSED) or not os.path.exists(TEST_PROCESSED):
        raise FileNotFoundError("Processed train/test not found. Run data_preprocessing.py first.")
    train_df = pd.read_csv(TRAIN_PROCESSED)
    test_df = pd.read_csv(TEST_PROCESSED)
    logging.info("Loaded processed splits: train=%d rows, test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator if denominator not in (0, None, math.nan) else 0.0
    except Exception:
        return 0.0


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature engineering is meaningful only if original columns exist; otherwise skip.
    required_cols = {
        "price_usd",
        "discount_offered_pct",
        "dev_budget_million_usd",
        "marketing_budget_million_usd",
        "units_sold_million",
        "positive_review_pct",
        "num_reviews",
        "avg_playtime_hours",
        "peak_concurrent_players",
        "story_length_hours",
        "dlc_count",
        "launch_bugs_reported",
        "patches_released",
        "awards_won",
    }

    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        logging.warning(
            "Skipping engineered feature creation because processed data lacks original columns. Missing: %s",
            missing,
        )
        return df

    def col(name, default=0):
        return df[name] if name in df.columns else default

    engineered_cols = []

    # Price after discount
    df["price_after_discount"] = col("price_usd") * (1 - col("discount_offered_pct") / 100)
    engineered_cols.append("price_after_discount")

    # Total budget
    df["total_budget_million_usd"] = col("dev_budget_million_usd") + col("marketing_budget_million_usd")
    engineered_cols.append("total_budget_million_usd")

    # Budget per unit sold (USD)
    df["budget_per_unit_usd"] = df.apply(
        lambda r: safe_divide(r.get("total_budget_million_usd", 0) * 1_000_000, r.get("units_sold_million", 0) * 1_000_000),
        axis=1,
    )
    engineered_cols.append("budget_per_unit_usd")

    # Effective positive reviews count (estimate)
    df["estimated_positive_reviews"] = col("positive_review_pct") / 100 * col("num_reviews")
    engineered_cols.append("estimated_positive_reviews")

    # Engagement score (simple product)
    df["engagement_score"] = col("avg_playtime_hours") * col("peak_concurrent_players")
    engineered_cols.append("engagement_score")

    # Value for money: story hours per dollar
    df["value_per_dollar"] = df.apply(
        lambda r: safe_divide(r.get("story_length_hours", 0), r.get("price_after_discount", 0)), axis=1
    )
    engineered_cols.append("value_per_dollar")

    # Content density: DLC per story hour
    df["dlc_per_story_hour"] = df.apply(
        lambda r: safe_divide(r.get("dlc_count", 0), r.get("story_length_hours", 0)), axis=1
    )
    engineered_cols.append("dlc_per_story_hour")

    # Bugs per patch
    df["bugs_per_patch"] = df.apply(
        lambda r: safe_divide(r.get("launch_bugs_reported", 0), (r.get("patches_released", 0) + 1)), axis=1
    )
    engineered_cols.append("bugs_per_patch")

    # Award efficiency: awards per million USD budget
    df["awards_per_budget_musd"] = df.apply(
        lambda r: safe_divide(r.get("awards_won", 0), max(r.get("total_budget_million_usd", 0), 1e-6)), axis=1
    )
    engineered_cols.append("awards_per_budget_musd")

    logging.info("Added engineered features: %s", engineered_cols)
    return df


def save_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    logging.info("Saving engineered train to %s", TRAIN_PREPROCESSED)
    train_df.to_csv(TRAIN_PREPROCESSED, index=False)
    logging.info("Saving engineered test to %s", TEST_PREPROCESSED)
    test_df.to_csv(TEST_PREPROCESSED, index=False)
    logging.info("Feature-engineered splits saved.")


def main() -> None:
    ensure_local_deps()
    configure_logging()
    logging.info("Starting feature engineering pipeline.")
    ensure_directories()
    train_proc, test_proc = load_processed()
    train_feat = add_features(train_proc)
    test_feat = add_features(test_proc)
    save_outputs(train_feat, test_feat)
    logging.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()
