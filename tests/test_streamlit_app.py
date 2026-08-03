import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "atc_agent.py"


class StreamlitComsolInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP_FILE), default_timeout=15).run()
        self.assertFalse(self.app.exception)

    def _number_input(self, label: str):
        return next(widget for widget in self.app.number_input if widget.label == label)

    def _checkbox(self, label: str):
        return next(widget for widget in self.app.checkbox if widget.label == label)

    def _button(self, label: str):
        return next(widget for widget in self.app.button if widget.label == label)

    def test_safe_defaults_and_control_example(self):
        self.assertEqual(self._number_input("Фактический ток, A").value, 400.0)
        self.assertEqual(self._number_input("Номинальный ток, A").value, 500.0)

        self._number_input("Фактический ток, A").set_value(300.0)
        self._button("Подставить контрольный пример COMSOL").click().run()

        self.assertEqual(self._number_input("Фактический ток, A").value, 400.0)
        self.assertAlmostEqual(
            self._number_input("Температура контакта, °C").value,
            53.41,
            places=2,
        )

    def test_extrapolation_requires_explicit_permission(self):
        self._number_input("Фактический ток, A").set_value(600.0)
        self._button("Рассчитать").click().run()

        messages = [item.value for item in self.app.info]
        self.assertTrue(any("не показано" in message for message in messages))
        self.assertFalse(
            any(
                item.value == "Сопоставление с физической моделью COMSOL"
                for item in self.app.subheader
            )
        )

        self._checkbox("Разрешить экстраполяцию COMSOL").set_value(True)
        self._button("Рассчитать").click().run()
        self.assertTrue(
            any(
                item.value == "Сопоставление с физической моделью COMSOL"
                for item in self.app.subheader
            )
        )


if __name__ == "__main__":
    unittest.main()
