import os
import unittest
from unittest.mock import patch

from app.settings import get_settings


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()

    def test_image_hosts_are_not_tied_to_a_test_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()

            settings = get_settings()

        self.assertEqual(settings.allowed_image_hosts, ())

    def test_image_host_allowlist_is_optional_configuration(self):
        with patch.dict(
            os.environ,
            {"ALLOWED_IMAGE_HOSTS": "images.example.com, media.example.com"},
            clear=True,
        ):
            get_settings.cache_clear()

            settings = get_settings()

        self.assertEqual(
            settings.allowed_image_hosts,
            ("images.example.com", "media.example.com"),
        )

    def test_cpu_default_uses_promptable_yoloe_checkpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()

            settings = get_settings()

        self.assertEqual(settings.yolo_model, "/data/models/yoloe-26n-seg.pt")
        self.assertEqual(settings.yolo_min_score, 0.25)
        self.assertEqual(settings.mask_min_component_area, 24)
        self.assertFalse(settings.enable_fba_refinement)
        self.assertEqual(settings.fba_edge_radius, 8)
        self.assertEqual(settings.fba_max_size, 768)
        self.assertEqual(settings.fba_min_iou, 0.75)
        self.assertFalse(settings.local_demo_enabled)

    def test_local_demo_can_be_enabled(self):
        with patch.dict(os.environ, {"ENABLE_LOCAL_DEMO": "true"}, clear=True):
            get_settings.cache_clear()

            settings = get_settings()

        self.assertTrue(settings.local_demo_enabled)
