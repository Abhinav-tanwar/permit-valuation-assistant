"""Train and save the notebook-compatible permit valuation pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predictor import (
    ARTIFACT_DIR,
    CATEGORICAL_SOURCE_COLUMNS,
    DATE_COLUMN,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_SOURCE_COLUMNS,
    PROJECT_DIR,
    REQUIRED_INPUT_COLUMNS,
    SQFT_COLUMNS,
    TARGET,
    PermitFeatureTransformer,
    _clean_numeric,
)


DATA_PATH = PROJECT_DIR / "data" / "permits.csv"
MODEL_VERSION = "1.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjusted_r2(y_true: pd.Series, y_pred: np.ndarray, feature_count: int) -> float:
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return float(1 - (1 - r2) * (n - 1) / (n - feature_count - 1))


def notebook_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def metric_set(
    y_log: pd.Series, pred_log: np.ndarray, feature_count: int
) -> Dict[str, float]:
    y_dollars = np.expm1(np.asarray(y_log))
    pred_dollars = np.maximum(np.expm1(pred_log), 0.0)
    return {
        "log_rmse": float(mean_squared_error(y_log, pred_log) ** 0.5),
        "log_mae": float(mean_absolute_error(y_log, pred_log)),
        "r2": float(r2_score(y_log, pred_log)),
        "adjusted_r2": adjusted_r2(y_log, pred_log, feature_count),
        "notebook_mape_percent": notebook_mape(y_log, pred_log),
        "dollar_rmse_against_capped_target": float(
            mean_squared_error(y_dollars, pred_dollars) ** 0.5
        ),
        "dollar_mae_against_capped_target": float(
            mean_absolute_error(y_dollars, pred_dollars)
        ),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("The curated permit CSV must be present.")

    original_hashes = {
        DATA_PATH.name: sha256(DATA_PATH),
    }
    df = pd.read_csv(DATA_PATH)
    required = REQUIRED_INPUT_COLUMNS + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    X = df[REQUIRED_INPUT_COLUMNS].copy()
    y_raw = _clean_numeric(df[TARGET])
    if y_raw.isna().any() or (y_raw < 0).any() or not np.isfinite(y_raw).all():
        raise ValueError("Target must contain finite, non-negative numeric values.")

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.30, random_state=42
    )
    target_cap = float(y_train_raw.quantile(0.99))
    y_train = np.log1p(y_train_raw.clip(upper=target_cap))
    y_test = np.log1p(y_test_raw.clip(upper=target_cap))

    numeric_engineered = [
        "Existing_SQFT_log",
        "Remodel_SQFT_log",
        "NewAdd_SQFT_log",
        "Dist_From_Center",
        "LatLon",
        "Time_Index",
    ]
    categorical_engineered = [
        "Permit Type",
        "Work Class",
        "Permit Class Mapped",
        "Number Of Floors_cat",
        "Housing Units_cat",
    ]
    one_hot = OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False)
    column_processor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric_engineered),
            ("categorical", one_hot, categorical_engineered),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [
            ("notebook_features", PermitFeatureTransformer()),
            ("encode", column_processor),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    feature_names = list(pipeline.named_steps["encode"].get_feature_names_out())
    transformer = pipeline.named_steps["notebook_features"]

    categories: Dict[str, Any] = {}
    for column in CATEGORICAL_SOURCE_COLUMNS:
        categories[column] = sorted(X_train[column].astype(str).unique().tolist())
    for column in ["Number Of Floors", "Housing Units"]:
        categories[column] = sorted(
            {
                PermitFeatureTransformer._category_string(value)
                for value in X_train[column].tolist()
            },
            key=float,
        )

    observed_ranges = {}
    numeric_frame = X_train.copy()
    for column in NUMERIC_SOURCE_COLUMNS:
        numeric_frame[column] = _clean_numeric(numeric_frame[column])
        observed_ranges[column] = {
            "min": float(numeric_frame[column].min()),
            "max": float(numeric_frame[column].max()),
        }
    applied_dates = pd.to_datetime(X_train[DATE_COLUMN], errors="coerce")
    observed_ranges[DATE_COLUMN] = {
        "min": applied_dates.min().date().isoformat(),
        "max": applied_dates.max().date().isoformat(),
    }

    timestamp = datetime.now(timezone.utc).isoformat()
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": timestamp,
        "model_type": "sklearn.linear_model.LinearRegression",
        "target": "log1p of upper-99th-percentile-capped Total Job Valuation",
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "random_state": 42,
        },
        "feature_count_after_encoding": len(feature_names),
        "train": metric_set(y_train, train_predictions, len(feature_names)),
        "test": metric_set(y_test, test_predictions, len(feature_names)),
        "metric_notes": {
            "log_errors": "RMSE and MAE are measured on the log1p target.",
            "dollar_errors": "Dollar errors compare with the target capped at the training 99th percentile.",
            "mape": "Notebook-compatible MAPE excludes rows whose log target is zero.",
        },
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": timestamp,
        "source_files_sha256": original_hashes,
        "required_input_columns": REQUIRED_INPUT_COLUMNS,
        "categories": categories,
        "observed_ranges": observed_ranges,
        "target_cap": target_cap,
        "feature_caps": transformer.sqft_caps_,
        "lat_mean": transformer.lat_mean_,
        "lon_mean": transformer.lon_mean_,
        "base_year": transformer.base_year_,
        "encoded_feature_names": feature_names,
    }
    artifact = {
        "pipeline": pipeline,
        "target_cap": target_cap,
        "feature_caps": transformer.sqft_caps_,
        "model_type": "LinearRegression",
        "model_version": MODEL_VERSION,
        "metrics": metrics,
        "metadata": metadata,
        "sklearn_version": sklearn.__version__,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reloaded = joblib.load(MODEL_PATH)
    np.testing.assert_allclose(
        reloaded["pipeline"].predict(X_test), test_predictions, rtol=0, atol=1e-12
    )
    for path, expected_hash in [
        (DATA_PATH, original_hashes[DATA_PATH.name]),
    ]:
        if sha256(path) != expected_hash:
            raise RuntimeError(
                f"Original source file changed during training: {path.name}"
            )

    print(f"Saved complete pipeline: {MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved metadata: {METADATA_PATH.relative_to(PROJECT_DIR)}")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
