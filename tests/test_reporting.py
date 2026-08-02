import os
import unittest
from unittest.mock import patch

from thermal_diagnostics.models import DiagnosticInputs
from thermal_diagnostics.physics import evaluate_contact
from thermal_diagnostics.reporting import enhance_report


class ReportingTests(unittest.TestCase):
    def test_missing_api_key_keeps_local_report(self):
        inputs = DiagnosticInputs(
            t_contact=40,
            t_reference=40,
            t_ambient=20,
            i_actual=800,
            i_nominal=800,
            temperature_uncertainty=0.1,
        )
        original = evaluate_contact(inputs)
        with patch.dict(os.environ, {}, clear=True):
            report, warning = enhance_report(original, inputs)
        self.assertEqual(report, original)
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
