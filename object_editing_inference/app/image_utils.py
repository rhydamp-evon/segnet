import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps


class ImageInputError(Exception):
    pass


def download_image(url, settings):
    parsed = urlparse(str(url))
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https"):
        raise ImageInputError("Only HTTP and HTTPS image URLs are supported.")
    if settings.allowed_image_hosts and hostname not in settings.allowed_image_hosts:
        raise ImageInputError(f"Image host is not allowed: {hostname}.")

    try:
        with requests.get(
            str(url),
            stream=True,
            timeout=settings.request_timeout,
            headers={"User-Agent": "AtlentisObjectEditing/1.0"},
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type and not content_type.startswith("image/"):
                raise ImageInputError("The source URL did not return an image.")

            data = bytearray()
            for chunk in response.iter_content(1024 * 1024):
                data.extend(chunk)
                if len(data) > settings.max_image_bytes:
                    raise ImageInputError(
                        "The source image is larger than the configured limit."
                    )
    except requests.RequestException as exc:
        raise ImageInputError(f"Could not download source image: {exc}") from exc

    return decode_image(bytes(data)), hashlib.sha256(data).hexdigest()


def decode_image(data):
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image
    except (OSError, ValueError) as exc:
        raise ImageInputError("The source image is invalid.") from exc


def image_pixel_hash(image):
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:{image.mode}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def safe_media_path(media_root, relative_path):
    path = (media_root / relative_path).resolve()
    try:
        path.relative_to(media_root)
    except ValueError as exc:
        raise ImageInputError("Invalid media path.") from exc
    return path


def save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    if not success:
        raise RuntimeError(f"Could not save mask: {path.name}")


def load_mask(media_root, relative_path, image_size):
    path = safe_media_path(media_root, relative_path)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ImageInputError(f"Mask does not exist: {relative_path}")
    width, height = image_size
    if mask.shape != (height, width):
        raise ImageInputError(f"Mask size does not match image: {relative_path}")
    return mask > 127


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_overlay(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 16:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) >= 3:
            polygons.append(points.astype(int).tolist())
    height, width = mask.shape
    return {
        "format": "polygons",
        "width": width,
        "height": height,
        "polygons": polygons,
    }


def save_result(image, media_root, relative_path):
    path = safe_media_path(media_root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path
