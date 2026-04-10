"""
Evaluation script for the video game success classifier.

Steps:
- Load best model from models/best_model.pkl
- Load preprocessed test split from data/preprocessed/test.csv
- Compute classification metrics (accuracy, precision, recall, f1, ROC AUC)
- Save metrics to artifact/metrics.json
- Save plots to artifact/:
    - roc_curve.png
    - precision_recall_curve.png
    - confusion_matrix.png
    - preds_vs_actual.png
- Log all steps to log/evaluation.txt and console
"""

import json
import logging
import os
import sys
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from utils import load_params, ensure_local_deps

# ----------------- config -----------------
PARAMS = load_params()
EVAL_PARAMS = PARAMS["evaluation"]

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, EVAL_PARAMS["preprocessed_subdir"])
TEST_PATH = os.path.join(PREPROCESSED_DIR, "test.csv")

MODEL_DIR = os.path.join(ROOT, "..", EVAL_PARAMS["model_dir"])
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

LOG_DIR = os.path.join(ROOT, "..", EVAL_PARAMS["log_dir"])
LOG_FILE = os.path.join(LOG_DIR, "evaluation.txt")

ARTIFACT_DIR = os.path.join(ROOT, "..", EVAL_PARAMS["artifact_dir"])

TARGET_COL = EVAL_PARAMS["target_col"]


def ensure_dependencies() -> None:
    """Ensure local .deps is on sys.path for offline installs."""
    ensure_local_deps()


def configure_logging() -> None:
    # ----------------- logging setup -----------------
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)


def ensure_directories() -> None:
    # ----------------- directories -----------------
    for path in (PREPROCESSED_DIR, MODEL_DIR, ARTIFACT_DIR, LOG_DIR):
        os.makedirs(path, exist_ok=True)
    logging.info("Ensured directories exist (data, model, artifact, log).")


def load_data_model() -> Tuple[pd.DataFrame, object]:
    # ----------------- load -----------------
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Test split not found. Run feature_enginnering.py and model_building.py first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found at models/best_model.pkl. Run model_building.py first.")

    logging.info("Loading test data from %s", TEST_PATH)
    test_df = pd.read_csv(TEST_PATH)
    logging.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    return test_df, model


def split_features_target(df: pd.DataFrame, target: str = TARGET_COL):
    # ----------------- split features/target -----------------
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' missing from test data.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    # ----------------- metrics -----------------
    is_multi = len(np.unique(y_true)) > 2
    avg = "weighted" if is_multi else "binary"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    if y_prob is not None:
        try:
            if is_multi:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = float("nan")
    logging.info("Metrics: %s", metrics)
    return metrics


def plot_roc(y_true, y_prob, path: str):
    # ----------------- plot roc -----------------
    classes = np.unique(y_true)
    if len(classes) > 2:
        y_true_bin = label_binarize(y_true, classes=classes)
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        title = "ROC Curve (micro-average)"
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        title = "ROC Curve"
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info("Saved ROC curve to %s", path)


def plot_pr(y_true, y_prob, path: str):
    # ----------------- plot pr -----------------
    classes = np.unique(y_true)
    if len(classes) > 2:
        y_true_bin = label_binarize(y_true, classes=classes)
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        title = "Precision-Recall Curve (micro-average)"
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        title = "Precision-Recall Curve"
    plt.figure()
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info("Saved Precision-Recall curve to %s", path)


def plot_confusion(y_true, y_pred, path: str):
    # ----------------- plot confusion -----------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info("Saved confusion matrix to %s", path)


def plot_preds_vs_actual(y_true, y_prob, path: str):
    # ----------------- plot preds vs actual -----------------
    classes = np.unique(y_true)
    if len(classes) > 2 and y_prob.ndim == 2:
        # Probability of true class for each sample
        true_indices = pd.Series(y_true).map({c: i for i, c in enumerate(classes)}).to_numpy()
        prob_true_class = y_prob[np.arange(len(y_prob)), true_indices]
        order = np.argsort(prob_true_class)
        sorted_prob = prob_true_class[order]
        sorted_true = np.ones_like(sorted_prob)  # actual is always 1 for true-class prob
        title = "True-class Probability (sorted)"
    else:
        order = np.argsort(y_prob)
        sorted_prob = y_prob[order]
        sorted_true = np.array(y_true)[order]
        title = "Predicted Probability vs Actual"
    plt.figure()
    plt.plot(sorted_prob, label="Predicted probability", color="tab:blue")
    plt.plot(sorted_true, label="Actual label", color="tab:orange", alpha=0.7)
    plt.xlabel("Samples (sorted by predicted prob)")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info("Saved preds-vs-actual plot to %s", path)


def save_metrics(metrics: Dict[str, float], path: str):
    # ----------------- save metrics -----------------
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved metrics to %s", path)


def main() -> None:
    # ----------------- main -----------------
    ensure_dependencies()
    configure_logging()
    logging.info("Starting evaluation pipeline.")
    ensure_directories()
    test_df, model = load_data_model()
    X_test, y_test = split_features_target(test_df)

    # Predict
    y_pred = model.predict(X_test)
    try:
        prob = model.predict_proba(X_test)
        # For binary, squeeze to 1D of positive class; for multi, keep 2D
        if prob.ndim == 2 and prob.shape[1] == 2:
            y_prob = prob[:, 1]
        else:
            y_prob = prob
    except Exception:
        y_prob = None
        logging.warning("Model does not support predict_proba; ROC/PR plots will be skipped.")

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Save metrics and plots
    save_metrics(metrics, os.path.join(ARTIFACT_DIR, "metrics.json"))

    if y_prob is not None:
        plot_roc(y_test, y_prob, os.path.join(ARTIFACT_DIR, "roc_curve.png"))
        plot_pr(y_test, y_prob, os.path.join(ARTIFACT_DIR, "precision_recall_curve.png"))
        plot_preds_vs_actual(y_test, y_prob, os.path.join(ARTIFACT_DIR, "preds_vs_actual.png"))
    else:
        logging.warning("Skipping probability-based plots due to missing predict_proba.")

    plot_confusion(y_test, y_pred, os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))

    logging.info("Evaluation completed.")


if __name__ == "__main__":
    main()
