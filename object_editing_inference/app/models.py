from functools import lru_cache
import importlib.util
import re
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

from app.settings import get_settings


class ModelConfigurationError(Exception):
    pass


YOLO_CATEGORY_MAP = {
    "bed": ("fixture.bed", "fixture"),
    "bench": ("fixture.bench", "fixture"),
    "chair": ("fixture.chair", "fixture"),
    "couch": ("fixture.sofa", "fixture"),
    "dining table": ("fixture.table", "fixture"),
    "microwave": ("fixture.microwave", "fixture"),
    "oven": ("fixture.oven", "fixture"),
    "refrigerator": ("fixture.refrigerator", "fixture"),
    "sink": ("fixture.sink", "fixture"),
    "toilet": ("fixture.toilet", "fixture"),
    "potted plant": ("object.plant", "object"),
    "tv": ("object.television", "object"),
}


def resolve_detector_backend(settings=None):
    settings = settings or get_settings()
    backend = settings.model_backend
    if backend not in {"auto", "sam3", "yolo"}:
        raise ModelConfigurationError(
            "MODEL_BACKEND must be auto, sam3, or yolo."
        )
    if backend == "auto":
        use_cuda = settings.device.startswith("cuda") and torch.cuda.is_available()
        return "sam3" if use_cuda else "yolo"
    return backend


def sam3_runtime_available(settings=None):
    settings = settings or get_settings()
    if not (
        settings.device.startswith("cuda")
        and torch.cuda.is_available()
        and importlib.util.find_spec("sam3") is not None
    ):
        return False
    if settings.sam3_checkpoint:
        return Path(settings.sam3_checkpoint).is_file()
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except (ImportError, OSError):
        return False


class Sam3Detector:
    def __init__(self, settings):
        if not settings.device.startswith("cuda") or not torch.cuda.is_available():
            raise ModelConfigurationError(
                "SAM 3 requires the CUDA object-editing service and an NVIDIA GPU."
            )

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        checkpoint_path = settings.sam3_checkpoint or None
        self.model = build_sam3_image_model(
            device=settings.device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
        )
        self.processor = Sam3Processor(
            self.model,
            device=settings.device,
            confidence_threshold=settings.sam3_min_score,
        )
        self.taxonomy = settings.taxonomy
        self.model_version = "sam3"
        self.lock = Lock()

    def detect(self, image, prompts=None):
        with self.lock:
            state = self.processor.set_image(image)
            detections = []
            categories = self.taxonomy
            if prompts:
                categories = [
                    {
                        "code": f"object.{_slug(prompt)}",
                        "label": prompt.strip().title(),
                        "level": "object",
                        "prompt": prompt,
                    }
                    for prompt in prompts
                ]
            for category in categories:
                output = self.processor.set_text_prompt(
                    prompt=category["prompt"], state=state
                )
                masks = self._numpy(output["masks"])
                scores = self._numpy(output["scores"])
                if masks.ndim == 4:
                    masks = masks[:, 0]
                elif masks.ndim == 2:
                    masks = masks[None, ...]

                for index, mask in enumerate(masks):
                    binary_mask = mask.astype(bool)
                    if not binary_mask.any():
                        continue
                    detections.append(
                        {
                            "category_code": category["code"],
                            "label": category["label"],
                            "level": category["level"],
                            "confidence": float(scores[index]),
                            "mask": binary_mask,
                        }
                    )
            return self._remove_duplicates(detections)

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _remove_duplicates(detections):
        kept = []
        for detection in sorted(
            detections, key=lambda item: item["confidence"], reverse=True
        ):
            duplicate = False
            for existing in kept:
                intersection = np.logical_and(
                    detection["mask"], existing["mask"]
                ).sum()
                union = np.logical_or(detection["mask"], existing["mask"]).sum()
                if union and intersection / union >= 0.9:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept


class YoloDetector:
    def __init__(self, settings, model_path=None):
        model_path = Path(model_path or settings.yolo_model)
        if model_path.parent != Path("."):
            model_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_promptable = model_path.name.lower().startswith("yoloe-")
        if self.is_promptable:
            from ultralytics import YOLOE

            self.model = YOLOE(str(model_path))
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
        self.device = settings.yolo_device
        self.min_score = settings.yolo_min_score
        self.mask_min_component_area = settings.mask_min_component_area
        self.model_version = model_path.stem
        self.taxonomy = settings.taxonomy
        self.taxonomy_by_prompt = {
            item["prompt"].strip().lower(): item for item in self.taxonomy
        }
        self.lock = Lock()

    def detect(self, image, prompts=None):
        with self.lock:
            active_prompts = prompts
            if self.is_promptable:
                active_prompts = prompts or [
                    item["prompt"] for item in self.taxonomy
                ]
                self.model.set_classes(active_prompts)
            elif prompts:
                raise ModelConfigurationError(
                    "Text prompts require a YOLOE *-seg.pt checkpoint."
                )
            results = self.model.predict(
                source=image,
                device=self.device,
                conf=self.min_score,
                retina_masks=True,
                verbose=False,
            )

        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.masks is None:
            return []

        height, width = image.height, image.width
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()
        detections = []
        for class_id, score, mask in zip(class_ids, scores, masks):
            label = str(result.names[class_id])
            taxonomy_item = self.taxonomy_by_prompt.get(label.lower())
            if taxonomy_item and not prompts:
                category_code = taxonomy_item["code"]
                display_label = taxonomy_item["label"]
                level = taxonomy_item["level"]
            else:
                category_code, level = self._category(label)
                display_label = label.replace("_", " ").title()
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_mask = _clean_instance_mask(
                mask > 0.5, self.mask_min_component_area
            )
            if binary_mask.any():
                detections.append(
                    {
                        "category_code": category_code,
                        "label": display_label,
                        "level": level,
                        "confidence": float(score),
                        "mask": binary_mask,
                    }
                )
        return detections

    @staticmethod
    def _category(label):
        if label in YOLO_CATEGORY_MAP:
            return YOLO_CATEGORY_MAP[label]
        return f"object.{_slug(label)}", "object"


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "item"


def _clean_instance_mask(mask, min_component_area=24):
    """Remove only tiny disconnected mask fragments without reshaping boundaries."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 2:
        return mask.astype(bool)
    cleaned = np.zeros_like(binary)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            cleaned[labels == component] = 1
    return cleaned.astype(bool) if cleaned.any() else mask.astype(bool)


class LamaInpainter:
    def __init__(self, settings):
        from simple_lama_inpainting import SimpleLama

        device_name = settings.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        self.model = SimpleLama(device=device)
        self.model_version = "simple-lama-0.1.2"
        self.lock = Lock()

    def inpaint(self, image, mask):
        with self.lock:
            return self.model(image, mask)


@lru_cache(maxsize=4)
def get_detector(model_path=None, backend_override=None):
    try:
        settings = get_settings()
        backend = backend_override or resolve_detector_backend(settings)
        if backend not in {"yolo", "sam3"}:
            raise ModelConfigurationError("Detector must be yolo or sam3.")
        if backend == "sam3":
            return Sam3Detector(settings)
        return YoloDetector(settings, model_path=model_path)
    except ModelConfigurationError:
        raise
    except Exception as exc:
        raise ModelConfigurationError(
            f"Could not load the {backend} detector: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_inpainter():
    try:
        return LamaInpainter(get_settings())
    except ModelConfigurationError:
        raise
    except Exception as exc:
        raise ModelConfigurationError(
            f"Could not load the LaMa model: {exc}"
        ) from exc
