# Object editing inference service

This service contains promptable YOLOE segmentation, optional SAM 3, and LaMa. It is separate from the
existing SkySegNet image so their Python and PyTorch versions do not conflict.

The default Compose file builds a CPU image with promptable YOLOE-26n-seg. The
GPU override builds the CUDA image with SAM 3. `MODEL_BACKEND=auto` selects the
detector available in the running image.

## Requirements

- CPU mode only requires Docker.
- SAM 3 mode requires an NVIDIA GPU, NVIDIA Container Toolkit, access to the
  gated `facebook/sam3` checkpoint, and a Hugging Face token.

Add these values to `src/.env`:

```env
OBJECT_EDITING_INFERENCE_URL=http://object-editing-gpu-service
OBJECT_EDITING_INFERENCE_API_KEY=replace-with-a-random-shared-key
OBJECT_EDITING_PUBLIC_MEDIA_BASE_URL=http://localhost:8080/media
OBJECT_EDITING_WEBHOOK_URL=
OBJECT_EDITING_WEBHOOK_API_KEY=
MODEL_BACKEND=auto
MODEL_DEVICE=cuda
YOLO_DEVICE=cpu
YOLO_MODEL=/data/models/yoloe-26n-seg.pt
ENABLE_FBA_REFINEMENT=true
FBA_REFINEMENT_MODEL=/data/legacy-checkpoints/final_fba.pth
```

On a CPU-only machine, no Hugging Face token is needed. For SAM 3, also add:

```env
HF_TOKEN=replace-with-your-hugging-face-token
```

For production, change `OBJECT_EDITING_PUBLIC_MEDIA_BASE_URL` to the public
media domain used by Atlentis.

`ALLOWED_IMAGE_HOSTS` is optional. Leave it unset for local testing. In
production, it can contain a comma-separated allowlist of the storage hosts
accepted for `source_image_url`.

## Start

The service is behind a Compose profile, so it does not change the normal Blue
Sky startup.

```bash
docker compose --profile object-editing build object-editing-gpu-service
docker compose --profile object-editing up -d object-editing-gpu-service
docker compose exec web python manage.py migrate
docker compose restart web celery
```

On an NVIDIA GPU host, expose the GPU with the override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile object-editing build object-editing-gpu-service
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile object-editing up -d object-editing-gpu-service
```

Check the service:

```bash
curl http://localhost:8001/health
```

## Local object-removal demo

The Compose service enables a standalone local demo that does not require
Django, PostgreSQL, Redis, Celery, or a webhook. Start only the inference
service:

```bash
docker compose --profile object-editing up --build object-editing-gpu-service
```

Then open `http://localhost:8001/demo`. Upload an image, enter one prompt or a
comma-separated list such as `bushes, trees`, choose Nano or Small, select any
returned instances, and choose Remove or Blur. Small is the demo default for
better accuracy; Nano remains available for a faster CPU comparison. The page
reports the actual detector and inference time after every scan.

When FBA refinement is enabled, only masks selected for editing are converted
into trimaps and passed through the existing `final_fba.pth` matting checkpoint.
The demo checkbox controls this per edit. Only the edge band may change.
Refined masks must remain close to the detector mask by area and IoU; otherwise
the service automatically keeps the original mask. FBA improves boundaries but
does not add object categories or recover a missed detection.

The demo also exposes SAM 3 in the detector selector when the CUDA image,
NVIDIA GPU, SAM 3 package, and gated checkpoint access are available. The CPU
image reports it as unavailable because the official SAM 3 runtime requires
CUDA; Nano and Small remain selectable there.

The first text-prompt request downloads the YOLOE checkpoint, the CLIP package,
and its text encoder. The first removal also downloads the LaMa checkpoint.
Later requests reuse the loaded models while the container is running.

Run its focused unit tests inside the built image:

```bash
docker compose --profile object-editing run --rm \
  object-editing-gpu-service python -m unittest discover -v
```

The first analysis downloads the selected detector checkpoint. The first
removal downloads the LaMa checkpoint, so both first requests take longer.
