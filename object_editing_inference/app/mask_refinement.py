from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from app.settings import get_settings


class FbaMaskRefiner:
    model_version = "fba-matting-final"

    def __init__(self, settings):
        checkpoint = Path(settings.fba_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FBA checkpoint does not exist: {checkpoint}")

        from core.networks.fba_matting import build_model

        args = SimpleNamespace(
            encoder="resnet50_GN_WS",
            decoder="fba_decoder",
            weights=str(checkpoint),
        )
        self.model = build_model(args)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.edge_radius = settings.fba_edge_radius
        self.max_size = settings.fba_max_size
        self.min_iou = settings.fba_min_iou
        self.lock = Lock()

    def refine(self, image, mask):
        ys, xs = np.where(mask)
        if not len(xs):
            return mask

        margin = max(24, self.edge_radius * 4)
        width, height = image.size
        x1 = max(0, int(xs.min()) - margin)
        y1 = max(0, int(ys.min()) - margin)
        x2 = min(width, int(xs.max()) + margin + 1)
        y2 = min(height, int(ys.max()) + margin + 1)

        image_crop = np.asarray(image, dtype=np.uint8)[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2].astype(bool)
        working_image, working_mask = _resize_for_refinement(
            image_crop, mask_crop, self.max_size
        )
        trimap = _build_trimap(working_mask, self.edge_radius)
        if not np.any(trimap[:, :, 1]):
            return mask

        alpha = self._predict_alpha(working_image, trimap)
        refined_working = alpha >= 0.5
        if refined_working.shape != mask_crop.shape:
            refined_crop = cv2.resize(
                refined_working.astype(np.uint8),
                (mask_crop.shape[1], mask_crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            refined_crop = refined_working

        if not _is_safe_refinement(mask_crop, refined_crop, self.min_iou):
            return mask

        refined = np.zeros_like(mask, dtype=bool)
        refined[y1:y2, x1:x2] = refined_crop
        return refined

    def _predict_alpha(self, image, trimap):
        from core.networks.fba_matting.transforms import (
            groupnorm_normalise_image,
            trimap_transform,
        )

        height, width = trimap.shape[:2]
        scaled_height = int(np.ceil(height / 8) * 8)
        scaled_width = int(np.ceil(width / 8) * 8)
        image_scaled = cv2.resize(
            image.astype(np.float32) / 255.0,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        trimap_scaled = cv2.resize(
            trimap.astype(np.float32),
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_NEAREST,
        )

        image_tensor = _to_tensor(image_scaled, self.device)
        trimap_tensor = _to_tensor(trimap_scaled, self.device)
        transformed_trimap = _to_tensor(
            trimap_transform(trimap_scaled).astype(np.float32), self.device
        )
        normalized_image = groupnorm_normalise_image(
            image_tensor.clone(), format="nchw"
        )

        with self.lock, torch.inference_mode():
            output = self.model(
                image_tensor,
                trimap_tensor,
                normalized_image,
                transformed_trimap,
            )
        alpha = output[0, 0].detach().cpu().numpy()
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LANCZOS4)
        alpha[trimap[:, :, 0] == 1] = 0
        alpha[trimap[:, :, 1] == 1] = 1
        return np.clip(alpha, 0, 1)


def _to_tensor(value, device):
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )


def _resize_for_refinement(image, mask, max_size):
    height, width = mask.shape
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return image, mask
    target = (max(8, round(width * scale)), max(8, round(height * scale)))
    resized_image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        mask.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return resized_image, resized_mask


def _build_trimap(mask, edge_radius):
    height, width = mask.shape
    radius = max(1, min(edge_radius, min(height, width) // 20))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = mask.astype(np.uint8)
    foreground = cv2.erode(binary, kernel, iterations=1)
    expanded = cv2.dilate(binary, kernel, iterations=1)
    trimap = np.zeros((height, width, 2), dtype=np.float32)
    trimap[:, :, 0] = expanded == 0
    trimap[:, :, 1] = foreground == 1
    return trimap


def _is_safe_refinement(original, refined, min_iou):
    original_area = int(original.sum())
    refined_area = int(refined.sum())
    if not original_area or not refined_area:
        return False
    area_ratio = refined_area / original_area
    intersection = np.logical_and(original, refined).sum()
    union = np.logical_or(original, refined).sum()
    iou = intersection / union if union else 0
    return 0.7 <= area_ratio <= 1.3 and iou >= min_iou


@lru_cache(maxsize=1)
def get_mask_refiner():
    return FbaMaskRefiner(get_settings())
