"""Prediction integrity tests using real rows from the Excel dataset."""

from __future__ import annotations

import copy
import math
import unittest

import joblib
import numpy as np
import pandas as pd

from predictor import (
    METRICS_PATH,
    MODEL_PATH,
    PROJECT_DIR,
    REQUIRED_INPUT_COLUMNS,
    explain_prediction,
    load_metadata,
    load_metrics,
    predict_permit_valuation,
    validate_inputs,
)


DATASET_PATH = PROJECT_DIR / "issue_permit_sample.xlsx"


def read_real_data():
    return (
        pd.read_excel(DATASET_PATH, engine="openpyxl")
        if DATASET_PATH.exists()
        else pd.read_csv(PROJECT_DIR / "data" / "permits.csv")
    )


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = read_real_data()
        cls.artifact = joblib.load(MODEL_PATH)
        cls.metadata = load_metadata()

    def real_record(self, index: int = 0):
        return self.df.iloc[index][REQUIRED_INPUT_COLUMNS].to_dict()

    def test_real_rows_are_finite_and_pipeline_matches(self):
        indices = [0, 2, 3, 10, 100, 1000, 4999]
        records = self.df.iloc[indices][REQUIRED_INPUT_COLUMNS]
        result = predict_permit_valuation(
            records, artifact=self.artifact, metadata=self.metadata
        )
        direct_log = self.artifact["pipeline"].predict(records)
        np.testing.assert_allclose(result["log_predictions"], direct_log)
        self.assertTrue(
            all(math.isfinite(v) and v >= 0 for v in result["dollar_predictions"])
        )

    def test_input_is_not_mutated(self):
        record = self.real_record()
        original = copy.deepcopy(record)
        predict_permit_valuation(record, self.artifact, self.metadata)
        self.assertEqual(record, original)

    def test_missing_input_rejected(self):
        record = self.real_record()
        record.pop("Latitude")
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_inputs(record, self.metadata)

    def test_negative_input_rejected(self):
        record = self.real_record()
        record["Total New Add SQFT"] = -1
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            validate_inputs(record, self.metadata)

    def test_nonfinite_inputs_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            record = self.real_record()
            record["Latitude"] = value
            with self.assertRaises(ValueError):
                validate_inputs(record, self.metadata)

    def test_unseen_category_rejected(self):
        record = self.real_record()
        record["Permit Type"] = "NOT_A_REAL_PERMIT"
        with self.assertRaisesRegex(ValueError, "not seen in training"):
            validate_inputs(record, self.metadata)

    def test_invalid_date_and_coordinate_rejected(self):
        record = self.real_record()
        record["Applied Date"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "valid date"):
            validate_inputs(record, self.metadata)
        record = self.real_record()
        record["Latitude"] = 91
        with self.assertRaisesRegex(ValueError, "between -90 and 90"):
            validate_inputs(record, self.metadata)

    def test_metrics_are_saved_with_artifact(self):
        metrics = load_metrics()
        self.assertEqual(metrics, self.artifact["metrics"])
        self.assertIn("test", metrics)
        self.assertIn("r2", metrics["test"])

    def test_ollama_cannot_supply_numeric_explanation(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "The valuation should be $999,999."}

        from unittest.mock import patch

        with patch("predictor.requests.post", return_value=FakeResponse()):
            result = explain_prediction(self.real_record())
        self.assertEqual(result["source"], "deterministic fallback")
        self.assertNotIn("$999", result["text"])


def print_real_row_comparison() -> None:
    df = read_real_data()
    metadata = load_metadata()
    sample = df.iloc[[0, 2, 3, 100, 1000]][REQUIRED_INPUT_COLUMNS]
    result = predict_permit_valuation(sample, metadata=metadata)
    actual = (
        df.iloc[[0, 2, 3, 100, 1000]]["Total Job Valuation"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .astype(float)
        .tolist()
    )
    print("\nReal dataset row comparison (diagnostic, not an accuracy assertion):")
    for index, actual_value, predicted in zip(
        sample.index, actual, result["dollar_predictions"]
    ):
        print(f"row={index} actual=${actual_value:,.2f} predicted=${predicted:,.2f}")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PredictionTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if outcome.wasSuccessful():
        print_real_row_comparison()
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
