from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

class AppTests(unittest.TestCase):
    def test_forecast_flow(self):
        self.assertTrue((ROOT / "app.py").exists(), "app must exist")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(m.label == "Giá dự đoán" for m in app.metric))
        if len(app.selectbox) > 1:
            app.selectbox[2].set_value("organic").run()
            self.assertFalse(any(m.label == "Giá dự đoán" for m in app.metric))
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(m.label == "Giá dự đoán" for m in app.metric))


    def test_model_choices_and_comparison(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        for choice in ["Keras", "So sánh cả hai"]:
            app.selectbox[0].set_value(choice).run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            forecasts = [m for m in app.metric if m.label.startswith("Giá dự đoán")]
            self.assertEqual(len(forecasts), 2 if choice == "So sánh cả hai" else 1)
        app.slider[0].set_value(10).run()
        self.assertTrue(any("±10%" in m.label for m in app.metric))

