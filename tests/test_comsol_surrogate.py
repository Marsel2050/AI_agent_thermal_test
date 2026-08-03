import csv
import unittest
from pathlib import Path

from thermal_diagnostics.comsol_surrogate import (
    FIT_MAX_ABS_ERROR_C,
    compare_with_comsol,
    describe_temperature_residual,
    estimate_contact_resistance,
    predict_max_temperature,
)


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "thermal_diagnostics"
    / "data"
    / "comsol_contact_sweep.csv"
)


class ComsolSurrogateTests(unittest.TestCase):
    def test_surrogate_reproduces_all_calibration_points(self):
        with DATA_FILE.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 25)
        errors = []
        for row in rows:
            predicted = predict_max_temperature(
                float(row["current_a"]),
                float(row["resistance_uohm"]),
            )
            errors.append(abs(predicted - float(row["temperature_c"])))

        self.assertLessEqual(max(errors), FIT_MAX_ABS_ERROR_C + 1e-12)

    def test_inverse_estimate_round_trips_prediction(self):
        temperature = predict_max_temperature(400.0, 20.0)
        estimated = estimate_contact_resistance(temperature, 400.0)
        self.assertAlmostEqual(estimated, 20.0, places=8)

    def test_zero_current_cannot_estimate_resistance(self):
        with self.assertRaises(ValueError):
            estimate_contact_resistance(20.0, 0.0)

    def test_comparison_marks_extrapolation_and_large_residual(self):
        comparison = compare_with_comsol(
            measured_temperature_c=200.0,
            current_a=600.0,
            assumed_resistance_uohm=150.0,
            temperature_uncertainty_c=2.0,
        )
        self.assertFalse(comparison.within_calibration_domain)
        self.assertGreaterEqual(len(comparison.warnings), 3)

    def test_negative_inverse_is_clipped_and_flagged(self):
        comparison = compare_with_comsol(
            measured_temperature_c=20.0,
            current_a=400.0,
            assumed_resistance_uohm=20.0,
        )
        self.assertEqual(comparison.estimated_resistance_uohm, 0.0)
        self.assertTrue(any("ограничена нулём" in item for item in comparison.warnings))

    def test_residual_description_explains_direction(self):
        self.assertEqual(
            describe_temperature_residual(-58.2),
            "Прогноз COMSOL выше измеренной температуры на 58.2 °C.",
        )
        self.assertEqual(
            describe_temperature_residual(4.6),
            "Измеренная температура выше прогноза COMSOL на 4.6 °C.",
        )
        self.assertIn("совпадает", describe_temperature_residual(0.01))


if __name__ == "__main__":
    unittest.main()
