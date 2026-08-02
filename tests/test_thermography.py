import unittest

import numpy as np

from thermal_diagnostics.thermography import (
    Region,
    region_from_drag,
    region_statistics,
    regions_overlap,
    thermal_preview,
)


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

    def test_drag_coordinates_are_scaled_to_temperature_matrix(self):
        region = region_from_drag(
            {
                "x1": 10,
                "y1": 20,
                "x2": 110,
                "y2": 120,
                "width": 320,
                "height": 256,
            },
            image_width=640,
            image_height=512,
        )
        self.assertEqual(region, Region(20, 40, 200, 200))

    def test_reverse_drag_is_bounded_to_image(self):
        region = region_from_drag(
            {
                "x1": 90,
                "y1": 80,
                "x2": -10,
                "y2": -20,
                "width": 100,
                "height": 100,
            },
            image_width=200,
            image_height=200,
        )
        self.assertEqual(region, Region(0, 0, 180, 160))

    def test_regions_overlap_but_touching_edges_do_not(self):
        first = Region(10, 10, 20, 20)
        self.assertTrue(regions_overlap(first, Region(25, 25, 10, 10)))
        self.assertFalse(regions_overlap(first, Region(30, 10, 10, 10)))


if __name__ == "__main__":
    unittest.main()
