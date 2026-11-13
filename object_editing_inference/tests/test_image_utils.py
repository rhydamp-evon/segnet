import tempfile
import unittest
from pathlib import Path

import numpy as np

from PIL import Image

from app.image_utils import (
    ImageInputError,
    image_pixel_hash,
    load_mask,
    safe_media_path,
    save_mask,
)


class ImageUtilsTests(unittest.TestCase):
    def test_masks_round_trip_without_changing_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            media_root = Path(directory).resolve()
            mask = np.zeros((12, 16), dtype=bool)
            mask[2:8, 4:11] = True
            path = safe_media_path(media_root, "masks/region.png")

            save_mask(mask, path)
            restored = load_mask(media_root, "masks/region.png", (16, 12))

            np.testing.assert_array_equal(restored, mask)

    def test_media_path_cannot_escape_media_root(self):
        with tempfile.TemporaryDirectory() as directory:
            media_root = Path(directory).resolve()

            with self.assertRaises(ImageInputError):
                safe_media_path(media_root, "../outside.png")

    def test_pixel_hash_is_stable_after_png_round_trip(self):
        image = Image.new("RGB", (8, 6), (20, 40, 60))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            image.save(path, format="PNG")
            with Image.open(path) as restored:
                restored = restored.convert("RGB")
                restored.load()

        self.assertEqual(image_pixel_hash(image), image_pixel_hash(restored))
