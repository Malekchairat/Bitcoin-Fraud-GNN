import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = "app.py"


class DashboardTests(unittest.TestCase):
    def run_app(self) -> AppTest:
        app = AppTest.from_file(APP_PATH, default_timeout=180)
        app.run(timeout=180)
        return app

    def test_dashboard_loads_without_exceptions(self) -> None:
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.metric), 5)

    def test_transaction_lookup_accepts_real_transaction(self) -> None:
        app = self.run_app()
        app.text_input[0].set_value("230425980")
        app.run(timeout=180)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_graph_controls_are_available(self) -> None:
        app = self.run_app()
        labels = [slider.label for slider in app.slider]
        self.assertIn("Maximum neighbors to display", labels)
        self.assertIn("Nodes in interactive overview", labels)
        self.assertIn("Relationships in interactive overview", labels)


if __name__ == "__main__":
    unittest.main()
