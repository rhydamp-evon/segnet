import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.models import (
    YoloDetector,
    _clean_instance_mask,
    _slug,
    resolve_detector_backend,
    sam3_runtime_available,
)


class DetectorBackendTests(unittest.TestCase):
    @patch("app.models.torch.cuda.is_available", return_value=False)
    def test_auto_uses_yolo_when_cuda_is_unavailable(self, is_available):
        settings = SimpleNamespace(model_backend="auto", device="cuda")

        self.assertEqual(resolve_detector_backend(settings), "yolo")
        is_available.assert_called_once_with()

    @patch("app.models.torch.cuda.is_available", return_value=True)
    def test_auto_uses_sam3_when_cuda_is_available(self, is_available):
        settings = SimpleNamespace(model_backend="auto", device="cuda")

        self.assertEqual(resolve_detector_backend(settings), "sam3")
        is_available.assert_called_once_with()

    @patch("app.models.torch.cuda.is_available", return_value=True)
    def test_cpu_device_keeps_auto_on_yolo(self, is_available):
        settings = SimpleNamespace(model_backend="auto", device="cpu")

        self.assertEqual(resolve_detector_backend(settings), "yolo")
        is_available.assert_not_called()

    def test_yolo_fixture_class_uses_fixture_hierarchy(self):
        self.assertEqual(
            YoloDetector._category("dining table"),
            ("fixture.table", "fixture"),
        )

    def test_unknown_yolo_class_uses_object_hierarchy(self):
        self.assertEqual(
            YoloDetector._category("wine glass"),
            ("object.wine_glass", "object"),
        )

    def test_prompt_is_converted_to_safe_category_slug(self):
        self.assertEqual(_slug("Road-side Trees"), "road_side_trees")

    def test_mask_cleanup_removes_only_small_disconnected_fragments(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:20, 5:20] = True
        mask[27, 27] = True

        cleaned = _clean_instance_mask(mask, min_component_area=8)

        self.assertTrue(cleaned[10, 10])
        self.assertFalse(cleaned[27, 27])

    @patch("app.models.importlib.util.find_spec", return_value=object())
    @patch("app.models.torch.cuda.is_available", return_value=True)
    @patch("app.models.Path.is_file", return_value=True)
    def test_sam3_is_available_only_with_cuda_package_and_checkpoint(
        self, is_file, cuda, find_spec
    ):
        settings = SimpleNamespace(
            device="cuda", sam3_checkpoint="/models/sam3.pt"
        )

        self.assertTrue(sam3_runtime_available(settings))
        cuda.assert_called_once_with()
        find_spec.assert_called_once_with("sam3")
        is_file.assert_called_once_with()

    @patch("app.models.torch.cuda.is_available", return_value=False)
    def test_sam3_is_not_available_on_cpu_service(self, cuda):
        settings = SimpleNamespace(device="cpu")

        self.assertFalse(sam3_runtime_available(settings))
        cuda.assert_not_called()
