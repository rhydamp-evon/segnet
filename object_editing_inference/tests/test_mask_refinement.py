import unittest

import numpy as np

from app.mask_refinement import _build_trimap, _is_safe_refinement


class MaskRefinementTests(unittest.TestCase):
    def test_trimap_has_background_unknown_edge_and_foreground(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True

        trimap = _build_trimap(mask, edge_radius=3)

        self.assertEqual(trimap.shape, (40, 40, 2))
        self.assertEqual(trimap[0, 0].tolist(), [1.0, 0.0])
        self.assertEqual(trimap[20, 20].tolist(), [0.0, 1.0])
        self.assertEqual(trimap[10, 10].tolist(), [0.0, 0.0])

    def test_similar_refined_mask_is_accepted(self):
        original = np.zeros((30, 30), dtype=bool)
        original[5:25, 5:25] = True
        refined = original.copy()
        refined[5, 5:8] = False

        self.assertTrue(_is_safe_refinement(original, refined, min_iou=0.75))

    def test_unrelated_refined_mask_is_rejected(self):
        original = np.zeros((30, 30), dtype=bool)
        original[2:12, 2:12] = True
        refined = np.zeros((30, 30), dtype=bool)
        refined[18:28, 18:28] = True

        self.assertFalse(_is_safe_refinement(original, refined, min_iou=0.75))


if __name__ == "__main__":
    unittest.main()
