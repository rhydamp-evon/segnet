import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.image_utils import ImageInputError, decode_image
from app.models import (
    ModelConfigurationError,
    resolve_detector_backend,
    sam3_runtime_available,
)
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DemoAnalyzeResponse,
    DemoEditRequest,
    DemoEditResponse,
    EditRequest,
    EditResponse,
)
from app.services import (
    analyze_demo_image,
    analyze_image,
    edit_demo_image,
    edit_image,
)
from app.settings import get_settings


app = FastAPI(title="Atlentis Object Editing Inference API", version="1.0.0")

DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"
DEMO_MEDIA_ROOT = get_settings().media_root / "object-editing-demo"
DEMO_PROMPT_ALIASES = {
    "table": ("table", "coffee table", "dining table", "side table", "desk"),
    "sofa": ("sofa", "couch"),
    "couch": ("couch", "sofa"),
    "plant": ("plant", "indoor plant", "houseplant"),
    "bush": ("bush", "shrub", "hedge"),
    "bushes": ("bush", "shrub", "hedge"),
    "tree": ("tree", "trees"),
    "trees": ("tree", "trees"),
    "road": ("road", "street", "driveway"),
}
DEMO_YOLO_MODELS = {
    "nano": "/data/models/yoloe-26n-seg.pt",
    "small": "/data/models/yoloe-26s-seg.pt",
}
DEMO_MODEL_VARIANTS = {*DEMO_YOLO_MODELS, "sam3"}
if get_settings().local_demo_enabled:
    DEMO_MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/demo-media",
        StaticFiles(directory=DEMO_MEDIA_ROOT, check_dir=False),
        name="demo-media",
    )


def verify_api_key(api_key: str | None = Header(default=None, alias="Api-Key")):
    configured_key = get_settings().api_key
    if configured_key and (
        not api_key or not secrets.compare_digest(api_key, configured_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


def verify_local_demo():
    if not get_settings().local_demo_enabled:
        raise HTTPException(status_code=404, detail="Local demo is disabled.")


def parse_prompts(value):
    requested = []
    seen = set()
    for item in value.split(","):
        prompt = " ".join(item.strip().split())
        key = prompt.lower()
        if prompt and key not in seen:
            requested.append(prompt)
            seen.add(key)
    if not requested:
        raise HTTPException(status_code=422, detail="Enter at least one prompt.")
    if len(requested) > 5:
        raise HTTPException(status_code=422, detail="Use at most five prompts.")
    if any(len(prompt) > 80 for prompt in requested):
        raise HTTPException(
            status_code=422, detail="Each prompt must be 80 characters or fewer."
        )

    prompts = []
    seen.clear()
    for prompt in requested:
        aliases = DEMO_PROMPT_ALIASES.get(prompt.lower(), (prompt,))
        for alias in aliases:
            key = alias.lower()
            if key not in seen:
                prompts.append(alias)
                seen.add(key)
    return prompts


@app.get("/health")
def health():
    settings = get_settings()
    backend = resolve_detector_backend(settings)
    device = settings.device if backend == "sam3" else settings.yolo_device
    return {
        "status": "ok",
        "configured_backend": settings.model_backend,
        "detector_backend": backend,
        "device": device,
        "local_demo_enabled": settings.local_demo_enabled,
        "demo_models": {
            "nano": True,
            "small": True,
            "sam3": sam3_runtime_available(settings),
        },
    }


@app.get("/demo", dependencies=[Depends(verify_local_demo)])
def local_demo():
    return FileResponse(DEMO_HTML_PATH, media_type="text/html")


@app.post(
    "/v1/demo/analyze",
    response_model=DemoAnalyzeResponse,
    dependencies=[Depends(verify_local_demo)],
)
def analyze_demo(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model_variant: str = Form("small"),
):
    settings = get_settings()
    data = image.file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413, detail="The uploaded image is too large."
        )
    try:
        if model_variant not in DEMO_MODEL_VARIANTS:
            raise HTTPException(
                status_code=422, detail="Model must be nano, small, or sam3."
            )
        if model_variant == "sam3" and not sam3_runtime_available(settings):
            raise HTTPException(
                status_code=503,
                detail=(
                    "SAM 3 is unavailable in this CPU service. Start the CUDA "
                    "object-editing image with an NVIDIA GPU and checkpoint access."
                ),
            )
        source = decode_image(data)
        prompts = parse_prompts(prompt)
        analysis_id = uuid4()
        detector_model = None
        detector_backend = "sam3" if model_variant == "sam3" else "yolo"
        if detector_backend == "yolo":
            detector_model = DEMO_YOLO_MODELS[model_variant]
        result = analyze_demo_image(
            source,
            analysis_id,
            prompts,
            detector_model=detector_model,
            detector_backend=detector_backend,
        )
        return {
            "analysis_id": analysis_id,
            "prompts": prompts,
            "model_variant": model_variant,
            "source_image_url": f"/demo-media/{analysis_id}/source.png",
            **result,
        }
    except ImageInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/v1/demo/edit",
    response_model=DemoEditResponse,
    dependencies=[Depends(verify_local_demo)],
)
def edit_demo(request: DemoEditRequest):
    try:
        return edit_demo_image(request, uuid4())
    except ImageInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(verify_api_key)],
)
def analyze(request: AnalyzeRequest):
    try:
        return analyze_image(request)
    except ImageInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/v1/edit",
    response_model=EditResponse,
    dependencies=[Depends(verify_api_key)],
)
def edit(request: EditRequest):
    try:
        return edit_image(request)
    except ImageInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
