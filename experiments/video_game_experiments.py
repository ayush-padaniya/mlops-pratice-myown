"""
Quick experiment script for the video game dataset.

Steps
- Load data from video_game_dataset.csv (target column: success_label)
- Basic cleaning (drop duplicates, simple NA imputation)
- Lightweight EDA prints
- Preprocess (one-hot encode categoricals, scale numerics)
- Train multiple classifiers (LogReg, RandomForest, GradientBoosting, XGBoost)
- Report accuracy for each on a train/test split

Note: Requires pandas, scikit-learn, xgboost. Install locally if missing:
    pip install --target ./.deps pandas scikit-learn
"""

import os
import sys
from typing import List, Tuple


def ensure_dependencies() -> None:
    """Fail fast with a friendly message if key packages are missing."""
    local_deps = os.path.join(os.path.dirname(__file__), "..", ".deps")
    if os.path.isdir(local_deps) and local_deps not in sys.path:
        sys.path.insert(0, local_deps)

    missing = []
    try:
        import pandas  # noqa: F401
    except Exception:
        missing.append("pandas")
    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")
    try:
        import xgboost  # noqa: F401
    except Exception:
        missing.append("xgboost")

    if missing:
        print(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall (local folder) with:\n"
            + "  pip install --target ./.deps "
            + " ".join(missing)
        )
        sys.exit(1)


def load_data(csv_path: str):
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def basic_clean(df):
    """Drop duplicates and fill missing values (median for numeric, mode for categorical)."""
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    df = df.drop_duplicates().copy()

    for col in df.columns:
        if col == "success_label":
            continue
        if is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            mode = df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else "missing"
            df[col] = df[col].fillna(fill_val)

    df["success_label"] = df["success_label"].fillna(df["success_label"].mode().iloc[0])
    return df


def run_eda(df) -> None:
    """Print a concise overview of the dataset."""
    import pandas as pd

    print("\n=== EDA ===")
    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}")

    # Target distribution
    print("\nTarget distribution (success_label):")
    print(df["success_label"].value_counts(dropna=False).head())

    # Missingness (top 10)
    miss = df.isna().mean().sort_values(ascending=False) * 100
    print("\nMissing values (% of rows) - top 10:")
    print(miss.head(10).round(2))

    # Basic numeric stats
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if num_cols:
        print("\nNumeric feature summary (first 5 cols):")
        print(df[num_cols].describe().transpose().head(5))

    # Sample records
    print("\nSample rows:")
    print(df.head(3))


def build_preprocessor(df):
    """Create ColumnTransformer with scaling and one-hot encoding."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    target = "success_label"
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    return ColumnTransformer(transformers), numeric_cols, categorical_cols


def get_models(random_state: int = 42):
    """Return a list of (name, estimator) pairs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier

    models = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, n_jobs=-1),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=random_state),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    return models


def train_and_evaluate(df) -> List[Tuple[str, float]]:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    target = "success_label"

    X = df.drop(columns=[target])
    y = df[target]

    # Encode labels to numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    results = []
    for name, model in get_models():
        clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))
        print(f"{name:>20}: accuracy = {acc:.4f}")

    return results


def main():
    ensure_dependencies()
    csv_path = os.path.join(os.path.dirname(__file__), "video_game_dataset.csv")
    df = load_data(csv_path)
    df = basic_clean(df)
    run_eda(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
