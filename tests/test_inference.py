import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("inference") is not None:
            import inference
            cls.api = inference
            cls.bundle = inference.load_bundle()
            cls.meta = cls.bundle["metadata"]
            cls.raw = pd.read_csv(ROOT / "data" / "prices.csv")
            cls.group = "SJC_Gold"
            cls.raw = cls.raw.rename(columns={"timestamp": "Date"})

    def setUp(self):
        self.assertTrue(hasattr(self, "api"), "inference module must exist")

    def test_saved_forecast(self):
        frame = self.api.prepare_data(self.raw, self.meta, self.group)
        actual = self.api.predict(frame[self.meta["target"]].tail(self.meta["lookback"]).tolist(), self.bundle, self.group)
        saved = pd.read_csv(ROOT / "model" / "next_step_forecast.csv")
        expected = saved.loc[saved.Model == "PyTorch RNN", "Prediction (triệu VND/lượng)"].iloc[0]
        self.assertAlmostEqual(actual, expected, delta=1e-4)

    def test_historical_predictions(self):
        saved = pd.read_csv(ROOT / "model" / "test_predictions.csv")
        for _, row in saved.groupby("group").first().reset_index().iterrows():
            frame = self.api.prepare_data(self.raw, self.meta, row["group"])
            window = frame.loc[frame.Date < pd.Timestamp(row.Date), self.meta["target"]].tail(self.meta["lookback"])
            actual = self.api.predict(window.tolist(), self.bundle, row["group"])
            self.assertAlmostEqual(actual, row["PyTorch RNN"], delta=1e-4)

    def test_keras_checkpoint_predictions(self):
        bundle = self.api.load_bundle("Keras")
        saved = pd.read_csv(ROOT / "model/test_predictions.csv")
        for _, row in saved.groupby("group").first().reset_index().iterrows():
            frame = self.api.prepare_data(self.raw, self.meta, row["group"])
            prices = frame.loc[frame.Date < pd.Timestamp(row.Date), self.meta["target"]].tail(self.meta["lookback"])
            actual = self.api.predict(prices.tolist(), bundle, row["group"])
            self.assertAlmostEqual(actual, row["Keras RNN"], delta=1e-4)
        frame = self.api.prepare_data(self.raw, self.meta, self.group)
        actual = self.api.predict(frame[self.meta["target"]].tail(self.meta["lookback"]).tolist(), bundle, self.group)
        saved = pd.read_csv(ROOT / "model/next_step_forecast.csv")
        self.assertAlmostEqual(actual, saved.loc[saved.Model == "Keras RNN", "Prediction (triệu VND/lượng)"].iloc[0], delta=1e-4)

    def test_accuracy_threshold_and_zero_actual(self):
        data = pd.DataFrame({"actual": [100., 100., 0.], "PyTorch RNN": [104., 110., 2.], "Keras RNN": [100., 103., 0.]})
        scores = self.api.evaluation_metrics(data, 5)
        self.assertEqual(scores["Đúng trong ngưỡng (%)"].tolist(), [50., 100.])
        self.assertEqual(scores["Mẫu tính tỉ lệ"].tolist(), [2, 2])
        self.assertEqual(self.api.evaluation_metrics(data, 10)["Đúng trong ngưỡng (%)"].tolist(), [100., 100.])

    def test_invalid_prices(self):
        n = self.meta["lookback"]
        for values in [[1.]*(n-1), [-1.]*n, [float("nan")]*n, [float("inf")]*n, [True]*n]:
            with self.subTest(values=values[:1]):
                with self.assertRaises(ValueError):
                    self.api.predict(values, self.bundle, self.group)
        with self.assertRaises(ValueError):
            self.api.predict([1.]*n, self.bundle, "unknown")

    def test_csv_validation_and_sorting(self):
        frame = self.api.prepare_data(self.raw, self.meta, self.group)
        self.assertTrue(frame.Date.is_monotonic_increasing)
        for bad in [self.raw.drop(columns=["Date"]), pd.concat([self.raw, self.raw]), self.raw.assign(Date="bad"),
                    self.raw.assign(**{self.meta["target"]: -1})]:
            with self.assertRaises(ValueError):
                self.api.prepare_data(bad, self.meta, self.group)
        with self.assertRaises(ValueError):
            self.api.prepare_data(frame.head(2), self.meta, self.group)


    def test_log_return_bundle(self):
        if not (ROOT / "model/log_return/metadata.json").exists():
            with self.assertRaises(FileNotFoundError):
                self.api.load_bundle("PyTorch", "log_return")
            return
        saved = pd.read_csv(ROOT / "model/log_return/test_predictions.csv")
        for framework in ("PyTorch", "Keras"):
            bundle = self.api.load_bundle(framework, "log_return")
            for _, row in saved.groupby("group").first().reset_index().iterrows():
                frame = self.api.prepare_data(self.raw, bundle["metadata"], row["group"])
                values = frame.loc[frame.Date < pd.Timestamp(row.Date), bundle["metadata"]["target"]].tail(bundle["metadata"]["lookback"]).tolist()
                result = self.api.predict(values, bundle, row["group"])
                self.assertAlmostEqual(result, row[framework + " RNN"], delta=1e-4)
                doubled = self.api.predict([v * 2 for v in values], bundle, row["group"])
                self.assertAlmostEqual(doubled, result * 2, delta=1e-4)
                with self.assertRaises(ValueError):
                    self.api.predict([0.] + values[1:], bundle, row["group"])

if __name__ == "__main__":
    unittest.main()

