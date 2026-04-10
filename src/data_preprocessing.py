"""
Data preprocessing pipeline for the video game dataset.

Steps:
1) Load raw train/test splits from data/raw (produced by data_ingestion.py).
2) Basic cleaning (duplicate drop + median/mode imputation).
3) Fit encoders/scalers on train, transform both splits.
4) Save processed splits to data/processed/train.csv and data/processed/test.csv.
5) Log every step to log/data_preprocessing.txt and console.
"""

import logging
import os
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_params, ensure_local_deps

# ----------------- config -----------------
PARAMS = load_params()
PREP_PARAMS = PARAMS["data_preprocessing"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, PARAMS["data_ingestion"]["raw_subdir"])
PROCESSED_DIR = os.path.join(DATA_DIR, PREP_PARAMS["processed_subdir"])
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", PREP_PARAMS["log_dir"])
LOG_FILE = os.path.join(LOG_DIR, "data_preprocessing.txt")

TARGET_COL = PREP_PARAMS["target_col"]

TRAIN_RAW = os.path.join(RAW_DIR, "train.csv")
TEST_RAW = os.path.join(RAW_DIR, "test.csv")
TRAIN_PROCESSED = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PROCESSED = os.path.join(PROCESSED_DIR, "test.csv")


def configure_logging() -> None:
    # ----------------- logging setup -----------------
    os.makedirs(LOG_DIR, exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format, handlers=handlers)


def ensure_directories() -> None:
    # ----------------- directories -----------------
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR):
        os.makedirs(path, exist_ok=True)
    logging.info("Ensured data directories exist.")


def load_raw_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ----------------- load -----------------
    if not os.path.exists(TRAIN_RAW) or not os.path.exists(TEST_RAW):
        raise FileNotFoundError("Raw train/test not found. Run data_ingestion.py first.")
    train_df = pd.read_csv(TRAIN_RAW)
    test_df = pd.read_csv(TEST_RAW)
    logging.info("Loaded raw splits: train=%d rows, test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = TARGET_COL) -> None:
    """Lightweight EDA logged only (no plots)."""
    # ----------------- eda -----------------
    logging.info("=== EDA: Shapes === train: %s, test: %s", train_df.shape, test_df.shape)

    # Target distribution
    if target in train_df.columns:
        logging.info("Target distribution (train):\n%s", train_df[target].value_counts(normalize=True).head())

    # Missingness
    train_miss = (train_df.isna().mean() * 100).sort_values(ascending=False)
    logging.info("Missing values %% (train) top 10:\n%s", train_miss.head(10).round(2))

    # Dtypes summary
    logging.info("Column dtypes:\n%s", train_df.dtypes)

    # Numeric summary (first few)
    num_cols = train_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if num_cols:
        logging.info("Numeric describe (first 5 cols):\n%s", train_df[num_cols].describe().transpose().head())

    # Sample rows
    logging.info("Sample train rows:\n%s", train_df.head(3))


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # ----------------- cleaning -----------------
    from pandas.api.types import is_numeric_dtype

    df = df.drop_duplicates().copy()
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")
    return df


def build_transformer(df: pd.DataFrame, target: str = TARGET_COL) -> ColumnTransformer:
    # ----------------- transformer -----------------
    from pandas.api.types import is_numeric_dtype

    numeric_cols = [c for c in df.columns if c != target and is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c != target and c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    ct = ColumnTransformer(transformers)
    logging.info(
        "Transformer built with %d numeric and %d categorical columns.",
        len(numeric_cols),
        len(categorical_cols),
    )
    return ct


def transform_split(
    ct: ColumnTransformer, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ----------------- transform -----------------
    y_train = train_df[target].reset_index(drop=True)
    y_test = test_df[target].reset_index(drop=True)

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    logging.info("Fitting transformer on train features.")
    X_train_t = ct.fit_transform(X_train)
    logging.info("Transforming test features.")
    X_test_t = ct.transform(X_test)

    # Build feature names via ColumnTransformer helper
    feature_names = ct.get_feature_names_out()

    train_data = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
    test_data = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

    train_processed = pd.DataFrame(train_data, columns=feature_names)
    test_processed = pd.DataFrame(test_data, columns=feature_names)

    # Append target back
    train_processed[target] = y_train.values
    test_processed[target] = y_test.values
    return train_processed, test_processed


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    # ----------------- save -----------------
    logging.info("Saving processed train to %s", TRAIN_PROCESSED)
    train_df.to_csv(TRAIN_PROCESSED, index=False)
    logging.info("Saving processed test to %s", TEST_PROCESSED)
    test_df.to_csv(TEST_PROCESSED, index=False)
    logging.info("Processed splits saved.")


def main() -> None:
    # ----------------- main -----------------
    ensure_local_deps()
    configure_logging()
    logging.info("Starting data preprocessing pipeline.")
    ensure_directories()
    train_raw, test_raw = load_raw_splits()
    run_eda(train_raw, test_raw)
    train_clean = basic_clean(train_raw)
    test_clean = basic_clean(test_raw)
    transformer = build_transformer(train_clean)
    train_processed, test_processed = transform_split(transformer, train_clean, test_clean)
    save_processed(train_processed, test_processed)
    logging.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
