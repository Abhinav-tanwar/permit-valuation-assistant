"""Training-compatible preprocessing and inference for permit valuations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, TransformerMixin


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "permit_valuation_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

TARGET = "Total Job Valuation"
DATE_COLUMN = "Applied Date"
NUMERIC_SOURCE_COLUMNS = [
    "Total Existing Bldg SQFT",
    "Remodel Repair SQFT",
    "Total New Add SQFT",
    "Number Of Floors",
    "Housing Units",
    "Latitude",
    "Longitude",
]
CATEGORICAL_SOURCE_COLUMNS = [
    "Permit Type",
    "Work Class",
    "Permit Class Mapped",
]
REQUIRED_INPUT_COLUMNS = (
    CATEGORICAL_SOURCE_COLUMNS + [DATE_COLUMN] + NUMERIC_SOURCE_COLUMNS
)
SQFT_COLUMNS = [
    "Total Existing Bldg SQFT",
    "Remodel Repair SQFT",
    "Total New Add SQFT",
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace("SF", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


class PermitFeatureTransformer(BaseEstimator, TransformerMixin):
    """Reproduce the final notebook feature engineering using fitted values."""

    output_columns = [
        "Existing_SQFT_log",
        "Remodel_SQFT_log",
        "NewAdd_SQFT_log",
        "Dist_From_Center",
        "LatLon",
        "Time_Index",
        "Permit Type",
        "Work Class",
        "Permit Class Mapped",
        "Number Of Floors_cat",
        "Housing Units_cat",
    ]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PermitFeatureTransformer":
        frame = self._coerce_frame(X)
        self.sqft_caps_ = {
            column: float(frame[column].quantile(0.99)) for column in SQFT_COLUMNS
        }
        self.lat_mean_ = float(frame["Latitude"].mean())
        self.lon_mean_ = float(frame["Longitude"].mean())
        self.base_year_ = int(frame[DATE_COLUMN].dt.year.min())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._coerce_frame(X)
        engineered = pd.DataFrame(index=frame.index)
        names = {
            "Total Existing Bldg SQFT": "Existing_SQFT_log",
            "Remodel Repair SQFT": "Remodel_SQFT_log",
            "Total New Add SQFT": "NewAdd_SQFT_log",
        }
        for source, destination in names.items():
            capped = frame[source].clip(upper=self.sqft_caps_[source])
            engineered[destination] = np.log1p(capped)

        engineered["Dist_From_Center"] = np.sqrt(
            (frame["Latitude"] - self.lat_mean_) ** 2
            + (frame["Longitude"] - self.lon_mean_) ** 2
        )
        engineered["LatLon"] = frame["Latitude"] * frame["Longitude"]
        engineered["Time_Index"] = (
            frame[DATE_COLUMN].dt.year - self.base_year_
        ) * 12 + frame[DATE_COLUMN].dt.month
        for column in CATEGORICAL_SOURCE_COLUMNS:
            engineered[column] = frame[column].astype(str)
        engineered["Number Of Floors_cat"] = frame["Number Of Floors"].map(
            self._category_string
        )
        engineered["Housing Units_cat"] = frame["Housing Units"].map(
            self._category_string
        )
        return engineered[self.output_columns]

    @staticmethod
    def _category_string(value: Any) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)

    @staticmethod
    def _coerce_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        missing = [
            column for column in REQUIRED_INPUT_COLUMNS if column not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        frame = X[REQUIRED_INPUT_COLUMNS].copy()
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
        for column in NUMERIC_SOURCE_COLUMNS:
            frame[column] = _clean_numeric(frame[column])
        return frame


def load_model(path: Path = MODEL_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {path}. Run: python train_model.py"
        )
    return joblib.load(path)


def load_metrics(path: Path = METRICS_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: Path = METADATA_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_inputs(
    records: Any, metadata: Optional[Mapping[str, Any]] = None
) -> pd.DataFrame:
    if isinstance(records, Mapping):
        frame = pd.DataFrame([dict(records)])
    elif isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Iterable):
        frame = pd.DataFrame(records)
    else:
        raise ValueError("Input must be a mapping, records, or a pandas DataFrame.")

    missing_columns = [
        column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required inputs: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("At least one input row is required.")
    if frame[REQUIRED_INPUT_COLUMNS].isna().any().any():
        raise ValueError("Inputs cannot contain missing values.")

    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("Applied Date must contain a valid date.")
    frame[DATE_COLUMN] = dates

    for column in NUMERIC_SOURCE_COLUMNS:
        values = _clean_numeric(frame[column])
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        frame[column] = values

    nonnegative = SQFT_COLUMNS + ["Number Of Floors", "Housing Units"]
    for column in nonnegative:
        if (frame[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")

    if ((frame["Latitude"] < -90) | (frame["Latitude"] > 90)).any():
        raise ValueError("Latitude must be between -90 and 90.")
    if ((frame["Longitude"] < -180) | (frame["Longitude"] > 180)).any():
        raise ValueError("Longitude must be between -180 and 180.")

    for column in CATEGORICAL_SOURCE_COLUMNS:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot be empty.")

    if metadata:
        categories = metadata.get("categories", {})
        category_fields = CATEGORICAL_SOURCE_COLUMNS + [
            "Number Of Floors",
            "Housing Units",
        ]
        for column in category_fields:
            allowed = {str(value) for value in categories.get(column, [])}
            actual = {
                PermitFeatureTransformer._category_string(value)
                if column in {"Number Of Floors", "Housing Units"}
                else str(value)
                for value in frame[column]
            }
            unknown = actual - allowed
            if unknown:
                raise ValueError(
                    f"{column} contains values not seen in training: "
                    + ", ".join(sorted(unknown))
                )
    return frame[REQUIRED_INPUT_COLUMNS]


def predict_permit_valuation(
    records: Any,
    artifact: Optional[Dict[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    artifact = artifact or load_model()
    metadata = metadata or load_metadata()
    frame = validate_inputs(records, metadata)
    log_predictions = artifact["pipeline"].predict(frame)
    dollar_predictions = np.maximum(np.expm1(log_predictions), 0.0)
    caps = artifact["feature_caps"]
    capped_fields = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        row_caps = [
            column
            for column in SQFT_COLUMNS
            if float(row[column]) > float(caps[column])
        ]
        capped_fields.append(row_caps)
    return {
        "log_predictions": [float(value) for value in log_predictions],
        "dollar_predictions": [float(value) for value in dollar_predictions],
        "capped_fields": capped_fields,
        "model_type": artifact["model_type"],
        "model_version": artifact["model_version"],
    }


def _fallback_explanation(record: Mapping[str, Any], capped: Iterable[str]) -> str:
    factors = [
        f"permit type {record['Permit Type']}",
        f"work class {record['Work Class']}",
        f"permit class {record['Permit Class Mapped']}",
        "the supplied project areas",
        "the project location",
        "and the application date",
    ]
    text = "The model considered " + ", ".join(factors) + "."
    if capped:
        text += " One or more project-area inputs exceeded the training cap."
    return text


def explain_prediction(
    record: Mapping[str, Any],
    capped_fields: Iterable[str] = (),
    ollama_url: str = "http://localhost:11434/api/generate",
    timeout: float = 20.0,
) -> Dict[str, str]:
    """Ask Ollama for factor-only prose; model valuation is never delegated."""
    fallback = _fallback_explanation(record, capped_fields)
    prompt = (
        "Explain, in two short sentences, which supplied permit attributes a trained "
        "regression model considered. Do not calculate, state, repeat, change, or "
        "suggest any valuation, price, currency amount, percentage, or other number. "
        "Do not claim causation. Discuss only the attributes in this JSON-like data: "
        f"{dict(record)}"
    )
    try:
        response = requests.post(
            ollama_url,
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        explanation = str(response.json().get("response", "")).strip()
        forbidden = re.compile(
            r"(?:[$€£¥]|\b\d+(?:[.,]\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|dollars?|euros?|pounds?|worth|price)\b)"
        )
        if not explanation or forbidden.search(explanation.lower()):
            return {"text": fallback, "source": "deterministic fallback"}
        return {"text": explanation, "source": "Ollama llama3.2"}
    except (requests.RequestException, ValueError, TypeError):
        return {"text": fallback, "source": "deterministic fallback"}
