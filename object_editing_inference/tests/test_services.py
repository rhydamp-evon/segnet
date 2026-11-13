import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.image_utils import save_mask
from app.services import _apply_blur, _build_regions, _combine_masks


class RegionHierarchyTests(unittest.TestCase):
    def test_regions_follow_structure_surface_fixture_object_hierarchy(self):
        floor_mask = np.zeros((100, 100), dtype=bool)
        floor_mask[50:100, :] = True
        table_mask = np.zeros((100, 100), dtype=bool)
        table_mask[45:80, 20:80] = True
        plate_mask = np.zeros((100, 100), dtype=bool)
        plate_mask[55:65, 45:55] = True
        detections = [
            self._detection("surface.floor", "Floor", "surface", floor_mask),
            self._detection("fixture.table", "Table", "fixture", table_mask),
            self._detection("object.plate", "Plate", "object", plate_mask),
        ]

        regions = _build_regions(detections, (100, 100))
        by_code = {region["category_code"]: region for region in regions}

        self.assertEqual(by_code["surface.floor"]["parent_id"], "room-1")
        self.assertEqual(
            by_code["fixture.table"]["parent_id"],
            by_code["surface.floor"]["region_id"],
        )
        self.assertEqual(
            by_code["object.plate"]["parent_id"],
            by_code["fixture.table"]["region_id"],
        )

    @staticmethod
    def _detection(code, label, level, mask):
        return {
            "category_code": code,
            "label": label,
            "level": level,
            "confidence": 0.9,
            "mask": mask,
        }


class BlurTests(unittest.TestCase):
    def test_blur_keeps_pixels_far_outside_selected_mask_unchanged(self):
        pixels = np.zeros((80, 80, 3), dtype=np.uint8)
        pixels[20:40, 20:40] = 255
        image = Image.fromarray(pixels)
        mask = np.zeros((80, 80), dtype=bool)
        mask[15:45, 15:45] = True

        result = np.asarray(_apply_blur(image, mask, strength=25))

        np.testing.assert_array_equal(result[70, 70], pixels[70, 70])
        self.assertFalse(np.array_equal(result[20, 20], pixels[20, 20]))


class SelectedMaskRefinementTests(unittest.TestCase):
    def test_refiner_runs_only_for_masks_passed_to_edit(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True

        class Refiner:
            def __init__(self):
                self.calls = 0

            def refine(self, image, source):
                self.calls += 1
                refined = source.copy()
                refined[4, 4] = True
                return refined

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_mask(mask, root / "mask.png")
            refiner = Refiner()
            combined, refined_count = _combine_masks(
                ["table-1"],
                {"table-1": SimpleNamespace(mask_path="mask.png")},
                root,
                (20, 20),
                image=Image.new("RGB", (20, 20)),
                refiner=refiner,
            )

        self.assertEqual(refiner.calls, 1)
        self.assertEqual(refined_count, 1)
        self.assertTrue(combined[4, 4])
