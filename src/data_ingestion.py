"""
Data ingestion script for the video game dataset.

Actions:
1) Download dataset from the provided URL.
2) Log each step for easy traceability.
3) Split into train and test sets (default 80/20).
4) Save splits under data/raw/train.csv and data/raw/test.csv (creates folders if missing).

No EDA or cleaning is performed here; this is ingestion only.
"""

import logging
import os
from typing import Tuple

import pandas as pd

from utils import load_params, ensure_local_deps


# Will be populated from params.yaml
PARAMS = load_params()
INGEST_PARAMS = PARAMS["data_ingestion"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", INGEST_PARAMS["data_dir"])
RAW_DIR = os.path.join(DATA_DIR, INGEST_PARAMS["raw_subdir"])
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", PARAMS["data_preprocessing"]["log_dir"])
LOG_FILE = os.path.join(LOG_DIR, "data_ingestion.txt")
TRAIN_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DIR, "test.csv")


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, LOG_DIR):
        os.makedirs(path, exist_ok=True)
    logging.info("Ensured data directories exist: %s", RAW_DIR)


def download_dataset(url: str = INGEST_PARAMS["url"]) -> pd.DataFrame:
    logging.info("Downloading dataset from %s", url)
    df = pd.read_csv(url)
    logging.info("Downloaded dataset with %d rows and %d columns", *df.shape)
    return df


def train_test_split_df(
    df: pd.DataFrame, test_size: float = INGEST_PARAMS["test_size"], seed: int = INGEST_PARAMS["random_state"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    logging.info("Splitting dataset: test_size=%.2f, random_state=%d", test_size, seed)
    test_df = df.sample(frac=test_size, random_state=seed)
    train_df = df.drop(test_df.index)
    logging.info("Split complete: train=%d rows, test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    logging.info("Saving train split to %s", TRAIN_PATH)
    train_df.to_csv(TRAIN_PATH, index=False)
    logging.info("Saving test split to %s", TEST_PATH)
    test_df.to_csv(TEST_PATH, index=False)
    logging.info("Finished saving splits.")


def main() -> None:
    ensure_local_deps()
    configure_logging()
    logging.info("Starting data ingestion pipeline.")
    ensure_directories()
    df = download_dataset()
    train_df, test_df = train_test_split_df(df)
    save_splits(train_df, test_df)
    logging.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()
