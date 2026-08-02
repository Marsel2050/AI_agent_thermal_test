import unittest

import numpy as np

from thermal_diagnostics.thermography import Region, region_statistics, thermal_preview


class ThermographyTests(unittest.TestCase):
    def test_region_statistics_and_percentile(self):
        matrix = np.arange(100, dtype=float).reshape(10, 10)
        stats = region_statistics(matrix, Region(2, 3, 4, 2))
        self.assertEqual(stats.minimum, 32.0)
        self.assertEqual(stats.maximum, 45.0)
        self.assertAlmostEqual(stats.median, 38.5)

    def test_region_is_bounded_to_image(self):
        self.assertEqual(
            Region(-2, 9, 20, 20).bounded(10, 10), Region(0, 9, 10, 1)
        )

    def test_preview_is_rgb_and_keeps_matrix_size(self):
        image = thermal_preview(
            np.arange(20, dtype=float).reshape(4, 5), Region(0, 0, 2, 2)
        )
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (5, 4))


if __name__ == "__main__":
    unittest.main()
