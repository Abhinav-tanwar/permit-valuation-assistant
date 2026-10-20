import sys, unittest, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from predictor import *
from unittest.mock import patch


class Integrity(unittest.TestCase):
    def test_training_only_centers_and_categories(self):
        d = pd.read_csv(PROJECT_DIR / "data/permits.csv")
        train, unused = train_test_split(d, test_size=0.3, random_state=42)
        m = load_metadata()
        self.assertAlmostEqual(m["lat_mean"], float(train.Latitude.mean()), places=10)
        for column in CATEGORICAL_SOURCE_COLUMNS:
            self.assertEqual(
                m["categories"][column],
                sorted(train[column].astype(str).unique().tolist()),
            )

    def test_spelled_out_currency_explanation_is_rejected(self):
        record = (
            pd.read_csv(PROJECT_DIR / "data/permits.csv")
            .iloc[0][REQUIRED_INPUT_COLUMNS]
            .to_dict()
        )

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "The estimate should be one million dollars."}

        with patch("predictor.requests.post", return_value=Response()):
            result = explain_prediction(record)
        self.assertEqual(result["source"], "deterministic fallback")


if __name__ == "__main__":
    unittest.main()
