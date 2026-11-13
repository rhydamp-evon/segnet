from collections import defaultdict
import logging
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from app.image_utils import (
    ImageInputError,
    download_image,
    image_pixel_hash,
    load_mask,
    mask_bbox,
    mask_overlay,
    safe_media_path,
    save_mask,
    save_result,
)
from app.settings import get_settings


logger = logging.getLogger(__name__)


PARENT_PREFERENCES = {
    "fixture.window": ("surface.wall",),
    "fixture.door": ("surface.wall",),
    "fixture.table": ("surface.floor",),
    "fixture.sofa": ("surface.floor",),
    "fixture.bed": ("surface.floor",),
    "fixture.chair": ("surface.floor",),
    "fixture.cabinet": ("surface.floor", "surface.wall"),
    "fixture.desk": ("surface.floor",),
    "object.plate": ("fixture.table", "fixture.desk"),
    "object.curtain": ("fixture.window",),
    "object.book": ("fixture.table", "fixture.desk", "fixture.cabinet"),
    "object.lamp": ("fixture.table", "fixture.desk", "fixture.cabinet"),
    "object.television": ("fixture.cabinet", "fixture.table"),
    "object.pillow": ("fixture.sofa", "fixture.bed"),
    "object.vase": ("fixture.table", "fixture.desk", "fixture.cabinet"),
}


def analyze_image(request):
    settings = get_settings()
    image, image_hash = download_image(request.source_image_url, settings)
    return _analyze_loaded_image(
        image=image,
        image_hash=image_hash,
        relative_root=f"object-editing/{request.analysis_id}",
    )


def analyze_demo_image(
    image, analysis_id, prompts, detector_model=None, detector_backend=None
):
    settings = get_settings()
    analysis_id = str(analysis_id)
    relative_root = f"object-editing-demo/{analysis_id}"
    save_result(image, settings.media_root, f"{relative_root}/source.png")
    return _analyze_loaded_image(
        image=image,
        image_hash=image_pixel_hash(image),
        relative_root=relative_root,
        prompts=prompts,
        detector_model=detector_model,
        detector_backend=detector_backend,
    )


def _analyze_loaded_image(
    image,
    image_hash,
    relative_root,
    prompts=None,
    detector_model=None,
    detector_backend=None,
):
    from app.models import get_detector

    settings = get_settings()
    detector = get_detector(detector_model, detector_backend)
    started_at = perf_counter()
    detections = detector.detect(image, prompts=prompts)
    inference_ms = round((perf_counter() - started_at) * 1000)
    regions = _build_regions(detections, image.size)

    for region in regions:
        relative_path = f"{relative_root}/masks/{region['region_id']}.png"
        path = safe_media_path(settings.media_root, relative_path)
        save_mask(region.pop("mask"), path)
        region["mask_path"] = relative_path

    return {
        "source_image_hash": image_hash,
        "model_version": detector.model_version,
        "inference_ms": inference_ms,
        "mask_refiner_version": None,
        "refined_regions": 0,
        "regions": regions,
    }


def edit_image(request):
    settings = get_settings()
    image, image_hash = download_image(request.source_image_url, settings)
    if image_hash != request.source_image_hash:
        raise ImageInputError(
            "The source image changed after object detection. Run analysis again."
        )

    result, model_version, refinement_model, refined_regions = _edit_loaded_image(
        image=image,
        regions=request.regions,
        remove_region_ids=request.remove_region_ids,
        blur_region_ids=request.blur_region_ids,
        blur_strength=request.blur_strength,
        refine_masks=request.refine_masks,
    )

    relative_path = (
        f"object-editing/{request.analysis_id}/results/{request.job_id}.png"
    )
    save_result(result, settings.media_root, relative_path)
    return {
        "result_image_url": (
            f"{settings.public_media_base_url}/{relative_path}"
        ),
        "model_version": model_version,
        "mask_refiner_version": refinement_model,
        "refined_regions": refined_regions,
    }


def edit_demo_image(request, job_id):
    settings = get_settings()
    analysis_id = str(request.analysis_id)
    relative_root = f"object-editing-demo/{analysis_id}"
    source_path = safe_media_path(
        settings.media_root, f"{relative_root}/source.png"
    )
    if not source_path.is_file():
        raise ImageInputError("The local demo source image does not exist.")

    try:
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            image.load()
    except (OSError, ValueError) as exc:
        raise ImageInputError("The local demo source image is invalid.") from exc

    if image_pixel_hash(image) != request.source_image_hash:
        raise ImageInputError(
            "The source image changed after object detection. Analyze it again."
        )

    expected_mask_prefix = f"{relative_root}/masks/"
    for region in request.regions:
        if not region.mask_path.startswith(expected_mask_prefix):
            raise ImageInputError("A selected mask does not belong to this analysis.")

    result, model_version, refinement_model, refined_regions = _edit_loaded_image(
        image=image,
        regions=request.regions,
        remove_region_ids=request.remove_region_ids,
        blur_region_ids=request.blur_region_ids,
        blur_strength=request.blur_strength,
        refine_masks=request.refine_masks,
    )
    result_relative_path = f"{relative_root}/results/{job_id}.png"
    save_result(result, settings.media_root, result_relative_path)
    return {
        "result_image_url": (
            f"/demo-media/{analysis_id}/results/{job_id}.png"
        ),
        "model_version": model_version,
        "mask_refiner_version": refinement_model,
        "refined_regions": refined_regions,
    }


def _edit_loaded_image(
    image,
    regions,
    remove_region_ids,
    blur_region_ids,
    blur_strength,
    refine_masks=False,
):
    settings = get_settings()
    regions_by_id = {region.region_id: region for region in regions}
    selected_ids = set(remove_region_ids + blur_region_ids)
    if not selected_ids:
        raise ImageInputError("Select at least one object to remove or blur.")
    unknown_ids = selected_ids - set(regions_by_id)
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise ImageInputError(f"Unknown selected region IDs: {unknown}.")
    conflicts = set(remove_region_ids).intersection(blur_region_ids)
    if conflicts:
        conflict_list = ", ".join(sorted(conflicts))
        raise ImageInputError(
            f"Objects cannot be removed and blurred together: {conflict_list}."
        )

    refiner = None
    refinement_model = None
    if refine_masks and settings.enable_fba_refinement:
        try:
            from app.mask_refinement import get_mask_refiner

            refiner = get_mask_refiner()
            refinement_model = refiner.model_version
        except Exception:
            logger.exception(
                "FBA mask refinement failed to load; using detector masks unchanged."
            )

    remove_mask, refined_remove = _combine_masks(
        remove_region_ids,
        regions_by_id,
        settings.media_root,
        image.size,
        image=image,
        refiner=refiner,
    )
    blur_mask, refined_blur = _combine_masks(
        blur_region_ids,
        regions_by_id,
        settings.media_root,
        image.size,
        image=image,
        refiner=refiner,
    )
    refined_regions = refined_remove + refined_blur

    result = image
    model_versions = []
    if remove_mask.any():
        kernel_size = max(1, settings.mask_dilation)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        remove_mask = cv2.dilate(
            remove_mask.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        mask_image = Image.fromarray(remove_mask.astype(np.uint8) * 255)
        from app.models import get_inpainter

        inpainter = get_inpainter()
        result = inpainter.inpaint(result, mask_image).convert("RGB")
        model_versions.append(inpainter.model_version)

    if blur_mask.any():
        result = _apply_blur(result, blur_mask, blur_strength)
        model_versions.append("opencv-gaussian-blur")
    return result, "+".join(model_versions), refinement_model, refined_regions


def _build_regions(detections, image_size):
    width, height = image_size
    root_mask = np.ones((height, width), dtype=bool)
    regions = [
        {
            "region_id": "room-1",
            "category_code": "structure.room",
            "label": "Room",
            "level": "structure",
            "parent_id": None,
            "confidence": 1.0,
            "bbox": [0, 0, width, height],
            "overlay": mask_overlay(root_mask),
            "mask": root_mask,
        }
    ]
    counters = defaultdict(int)

    for detection in detections:
        code = detection["category_code"]
        counters[code] += 1
        region_id = f"{code.replace('.', '-')}-{counters[code]}"
        regions.append(
            {
                "region_id": region_id,
                "category_code": code,
                "label": detection["label"],
                "level": detection["level"],
                "parent_id": None,
                "confidence": detection["confidence"],
                "bbox": mask_bbox(detection["mask"]),
                "overlay": mask_overlay(detection["mask"]),
                "mask": detection["mask"],
            }
        )

    for region in regions[1:]:
        if region["level"] == "surface":
            region["parent_id"] = "room-1"
        elif region["level"] == "fixture":
            surfaces = [item for item in regions if item["level"] == "surface"]
            parent = _best_parent(region, surfaces)
            region["parent_id"] = parent["region_id"] if parent else "room-1"
        elif region["level"] == "object":
            fixtures = [item for item in regions if item["level"] == "fixture"]
            parent = _best_parent(region, fixtures)
            region["parent_id"] = parent["region_id"] if parent else "room-1"
    return regions


def _best_parent(region, candidates):
    preferred_codes = PARENT_PREFERENCES.get(region["category_code"], ())
    preferred = [
        candidate
        for candidate in candidates
        if candidate["category_code"] in preferred_codes
    ]
    pool = preferred or candidates
    scored = [
        (_parent_score(region["bbox"], candidate["bbox"]), candidate)
        for candidate in pool
        if candidate["bbox"]
    ]
    scored = [item for item in scored if item[0] > 0]
    return max(scored, default=(0, None), key=lambda item: item[0])[1]


def _parent_score(child_bbox, parent_bbox):
    if not child_bbox or not parent_bbox:
        return 0
    child_x1, child_y1, child_x2, child_y2 = child_bbox
    parent_x1, parent_y1, parent_x2, parent_y2 = parent_bbox
    child_center_x = (child_x1 + child_x2) / 2
    child_center_y = (child_y1 + child_y2) / 2
    parent_width = parent_x2 - parent_x1
    parent_height = parent_y2 - parent_y1
    margin_x = parent_width * 0.15
    margin_y = parent_height * 0.15
    center_inside = (
        parent_x1 - margin_x <= child_center_x <= parent_x2 + margin_x
        and parent_y1 - margin_y <= child_center_y <= parent_y2 + margin_y
    )

    intersection_width = max(0, min(child_x2, parent_x2) - max(child_x1, parent_x1))
    intersection_height = max(0, min(child_y2, parent_y2) - max(child_y1, parent_y1))
    intersection = intersection_width * intersection_height
    child_area = max(1, (child_x2 - child_x1) * (child_y2 - child_y1))
    return intersection / child_area + (1.0 if center_inside else 0.0)


def _combine_masks(
    region_ids, regions, media_root, image_size, image=None, refiner=None
):
    width, height = image_size
    combined = np.zeros((height, width), dtype=bool)
    refined_regions = 0
    for region_id in region_ids:
        mask = load_mask(
            media_root, regions[region_id].mask_path, image_size
        )
        if refiner is not None and image is not None:
            refined = refiner.refine(image, mask)
            if not np.array_equal(mask, refined):
                mask = refined
                refined_regions += 1
        combined |= mask
    return combined, refined_regions


def _apply_blur(image, mask, strength):
    image_array = np.asarray(image, dtype=np.uint8)
    sigma = max(1.0, strength / 4.0)
    kernel_size = max(3, int(sigma * 6) | 1)
    blurred = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), sigma)
    feathered_mask = cv2.GaussianBlur(
        mask.astype(np.float32), (0, 0), max(1.0, sigma / 2.0)
    )
    alpha = np.clip(feathered_mask[..., None], 0.0, 1.0)
    result = image_array * (1.0 - alpha) + blurred * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
