import unittest

from thermal_diagnostics.models import AssessmentStatus, DiagnosticInputs
from thermal_diagnostics.physics import evaluate_contact


def make_inputs(**changes):
    values = {
        "t_contact": 40.0,
        "t_reference": 40.0,
        "t_ambient": 20.0,
        "i_actual": 800.0,
        "i_nominal": 800.0,
        "temperature_uncertainty": 0.1,
    }
    values.update(changes)
    return DiagnosticInputs(**values)


class PhysicsTests(unittest.TestCase):
    def test_low_load_is_not_reported_as_normal(self):
        result = evaluate_contact(make_inputs(i_actual=200.0))
        self.assertEqual(result.status, AssessmentStatus.NOT_ASSESSABLE)
        self.assertFalse(result.assessable)
        self.assertIsNone(result.k_defect)

    def test_defect_coefficient_uses_healthy_reference_rise(self):
        result = evaluate_contact(make_inputs(t_contact=50.0, t_reference=35.0))
        self.assertAlmostEqual(result.k_defect, 2.0)
        self.assertEqual(result.status, AssessmentStatus.EMERGENCY)
        self.assertTrue(result.requires_human_approval)

    def test_equal_rises_are_normal_when_measurement_is_precise(self):
        result = evaluate_contact(make_inputs())
        self.assertAlmostEqual(result.k_defect, 1.0)
        self.assertEqual(result.status, AssessmentStatus.NORMAL)

    def test_reference_rise_near_uncertainty_is_rejected(self):
        result = evaluate_contact(
            make_inputs(t_reference=21.0, temperature_uncertainty=2.0)
        )
        self.assertEqual(result.status, AssessmentStatus.NOT_ASSESSABLE)

    def test_passport_temperature_limit_can_trigger_emergency(self):
        result = evaluate_contact(
            make_inputs(t_contact=44.0, t_reference=41.0, max_permissible_rise=20.0)
        )
        self.assertEqual(result.status, AssessmentStatus.EMERGENCY)


if __name__ == "__main__":
    unittest.main()
