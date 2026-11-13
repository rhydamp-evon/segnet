import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_TAXONOMY = [
    {"code": "surface.floor", "label": "Floor", "level": "surface", "prompt": "floor"},
    {"code": "surface.wall", "label": "Wall", "level": "surface", "prompt": "wall"},
    {
        "code": "surface.ceiling",
        "label": "Ceiling",
        "level": "surface",
        "prompt": "ceiling",
    },
    {"code": "fixture.table", "label": "Table", "level": "fixture", "prompt": "table"},
    {"code": "fixture.sofa", "label": "Sofa", "level": "fixture", "prompt": "sofa"},
    {
        "code": "fixture.window",
        "label": "Window",
        "level": "fixture",
        "prompt": "window",
    },
    {"code": "fixture.bed", "label": "Bed", "level": "fixture", "prompt": "bed"},
    {"code": "fixture.chair", "label": "Chair", "level": "fixture", "prompt": "chair"},
    {"code": "fixture.door", "label": "Door", "level": "fixture", "prompt": "door"},
    {
        "code": "fixture.cabinet",
        "label": "Cabinet",
        "level": "fixture",
        "prompt": "cabinet",
    },
    {"code": "fixture.desk", "label": "Desk", "level": "fixture", "prompt": "desk"},
    {"code": "object.plate", "label": "Plate", "level": "object", "prompt": "plate"},
    {
        "code": "object.curtain",
        "label": "Curtain",
        "level": "object",
        "prompt": "curtain",
    },
    {"code": "object.book", "label": "Book", "level": "object", "prompt": "book"},
    {"code": "object.lamp", "label": "Lamp", "level": "object", "prompt": "lamp"},
    {
        "code": "object.television",
        "label": "Television",
        "level": "object",
        "prompt": "television",
    },
    {
        "code": "object.plant",
        "label": "Plant",
        "level": "object",
        "prompt": "indoor plant",
    },
    {"code": "object.pillow", "label": "Pillow", "level": "object", "prompt": "pillow"},
    {"code": "object.vase", "label": "Vase", "level": "object", "prompt": "vase"},
    {"code": "object.rug", "label": "Rug", "level": "object", "prompt": "rug"},
]


@dataclass(frozen=True)
class Settings:
    media_root: Path
    public_media_base_url: str
    api_key: str
    model_backend: str
    device: str
    sam3_checkpoint: str
    sam3_min_score: float
    yolo_model: str
    yolo_device: str
    yolo_min_score: float
    mask_min_component_area: int
    enable_fba_refinement: bool
    fba_checkpoint: str
    fba_edge_radius: int
    fba_max_size: int
    fba_min_iou: float
    request_timeout: int
    max_image_bytes: int
    mask_dilation: int
    allowed_image_hosts: tuple
    taxonomy: list
    local_demo_enabled: bool


@lru_cache(maxsize=1)
def get_settings():
    taxonomy_value = os.environ.get("OBJECT_EDITING_TAXONOMY", "")
    taxonomy = json.loads(taxonomy_value) if taxonomy_value else DEFAULT_TAXONOMY
    allowed_hosts = tuple(
        host.strip().lower()
        for host in os.environ.get("ALLOWED_IMAGE_HOSTS", "").split(",")
        if host.strip()
    )
    return Settings(
        media_root=Path(os.environ.get("MEDIA_ROOT", "/data/media")).resolve(),
        public_media_base_url=os.environ.get(
            "OBJECT_EDITING_PUBLIC_MEDIA_BASE_URL",
            os.environ.get("PUBLIC_MEDIA_BASE_URL", "http://localhost:8080/media"),
        ).rstrip("/"),
        api_key=os.environ.get(
            "OBJECT_EDITING_INFERENCE_API_KEY",
            os.environ.get("INFERENCE_API_KEY", ""),
        ),
        model_backend=os.environ.get("MODEL_BACKEND", "auto").lower(),
        device=os.environ.get("MODEL_DEVICE", "cuda"),
        sam3_checkpoint=os.environ.get("SAM3_CHECKPOINT", ""),
        sam3_min_score=float(os.environ.get("SAM3_MIN_SCORE", "0.5")),
        yolo_model=os.environ.get(
            "YOLO_MODEL", "/data/models/yoloe-26n-seg.pt"
        ),
        yolo_device=os.environ.get("YOLO_DEVICE", "cpu"),
        yolo_min_score=float(os.environ.get("YOLO_MIN_SCORE", "0.25")),
        mask_min_component_area=int(
            os.environ.get("MASK_MIN_COMPONENT_AREA", "24")
        ),
        enable_fba_refinement=os.environ.get(
            "ENABLE_FBA_REFINEMENT", "false"
        ).lower() in {"1", "true", "yes"},
        fba_checkpoint=os.environ.get(
            "FBA_REFINEMENT_MODEL", "/data/legacy-checkpoints/final_fba.pth"
        ),
        fba_edge_radius=int(os.environ.get("FBA_EDGE_RADIUS", "8")),
        fba_max_size=int(os.environ.get("FBA_MAX_SIZE", "768")),
        fba_min_iou=float(os.environ.get("FBA_MIN_IOU", "0.75")),
        request_timeout=int(os.environ.get("IMAGE_REQUEST_TIMEOUT", "30")),
        max_image_bytes=int(os.environ.get("MAX_IMAGE_BYTES", str(30 * 1024 * 1024))),
        mask_dilation=int(os.environ.get("MASK_DILATION", "7")),
        allowed_image_hosts=allowed_hosts,
        taxonomy=taxonomy,
        local_demo_enabled=os.environ.get(
            "ENABLE_LOCAL_DEMO", "false"
        ).lower() in {"1", "true", "yes"},
    )
