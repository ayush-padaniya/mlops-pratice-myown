"""
Model building script for the video game dataset.

Steps:
- Load preprocessed train/test splits from data/preprocessed (output of feature_enginnering.py).
- Train candidate classifiers (Logistic Regression + optional stronger models).
- Select the best model achieving >= 0.97 accuracy (measured on the provided test split).
- Save the best model to models/best_model.pkl.
- Log all steps to log/model_building.txt and console.

Note: This script performs only training and model selection; a separate script can handle full evaluation/metrics reporting.
"""

import logging
import os
import sys
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from utils import load_params, ensure_local_deps

# ----------------- config -----------------
PARAMS = load_params()
MB_PARAMS = PARAMS["model_building"]

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, MB_PARAMS["preprocessed_subdir"])
TRAIN_PATH = os.path.join(PREPROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PREPROCESSED_DIR, "test.csv")

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", MB_PARAMS["log_dir"])
LOG_FILE = os.path.join(LOG_DIR, "model_building.txt")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", MB_PARAMS["model_dir"])
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
TARGET_COL = MB_PARAMS["target_col"]
ACC_THRESHOLD = MB_PARAMS.get("accuracy_threshold", 0.97)
RANDOM_STATE = MB_PARAMS.get("random_state", 42)


def ensure_dependencies() -> None:
    """Ensure ./.deps is on path if present for local installs."""
    ensure_local_deps()


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
    for path in (PREPROCESSED_DIR, LOG_DIR, MODEL_DIR):
        os.makedirs(path, exist_ok=True)
    logging.info("Ensured required directories exist.")


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ----------------- load -----------------
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Preprocessed train/test not found. Run feature_enginnering.py first.")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    logging.info("Loaded preprocessed splits: train=%d rows, test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


def split_features_target(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = TARGET_COL):
    # ----------------- split features/target -----------------
    if target not in train_df.columns:
        raise KeyError(f"Target column '{target}' not found in train data.")
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target] if target in test_df.columns else None
    return X_train, y_train, X_test, y_test


def get_models(random_state: int = RANDOM_STATE):
    # ----------------- model zoo -----------------
    cfg = MB_PARAMS.get("models", {})

    lr_cfg = cfg.get("logistic_regression", {})
    rf_cfg = cfg.get("random_forest", {})
    gb_cfg = cfg.get("gradient_boosting", {})

    lr = LogisticRegression(
        max_iter=lr_cfg.get("max_iter", 200),
        n_jobs=lr_cfg.get("n_jobs", -1),
        penalty=lr_cfg.get("penalty", "l2"),
        C=lr_cfg.get("C", 1.0),
        solver=lr_cfg.get("solver", "lbfgs"),
    )

    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 400),
        max_depth=rf_cfg.get("max_depth", None),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=rf_cfg.get("random_state", random_state),
    )

    gb = GradientBoostingClassifier(
        n_estimators=gb_cfg.get("n_estimators", 100),
        learning_rate=gb_cfg.get("learning_rate", 0.1),
        max_depth=gb_cfg.get("max_depth", 3),
        random_state=gb_cfg.get("random_state", random_state),
    )

    return [
        ("LogisticRegression", lr),
        ("RandomForest", rf),
        ("GradientBoosting", gb),
    ]


def train_and_select(X_train, y_train, X_test, y_test) -> Tuple[str, object, float]:
    # ----------------- training -----------------
    best_name, best_model, best_acc = None, None, -1.0
    for name, model in get_models():
        logging.info("Training model: %s", name)
        model.fit(X_train, y_train)
        if y_test is not None:
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
        else:
            # Fallback to training accuracy if test not available
            acc = accuracy_score(y_train, model.predict(X_train))
        logging.info("%s accuracy: %.4f", name, acc)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model
    logging.info("Best model: %s (accuracy=%.4f)", best_name, best_acc)

    if best_acc < ACC_THRESHOLD:
        logging.warning(
            "Best model accuracy %.4f is below desired threshold %.2f. "
            "Consider revisiting features/hyperparameters.",
            best_acc,
            ACC_THRESHOLD,
        )
    return best_name, best_model, best_acc


def save_model(model) -> None:
    # ----------------- save -----------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logging.info("Saved best model to %s", MODEL_PATH)


def main() -> None:
    # ----------------- main -----------------
    ensure_dependencies()
    configure_logging()
    logging.info("Starting model building pipeline.")
    ensure_directories()
    train_df, test_df = load_splits()
    X_train, y_train, X_test, y_test = split_features_target(train_df, test_df)
    best_name, best_model, best_acc = train_and_select(X_train, y_train, X_test, y_test)
    save_model(best_model)
    logging.info("Model building completed. Best: %s (acc=%.4f)", best_name, best_acc)


if __name__ == "__main__":
    main()
