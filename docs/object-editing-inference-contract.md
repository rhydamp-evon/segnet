# Object editing inference contract

The Django application deliberately does not import SAM, LaMa, or another GPU
model. The independent implementation is in `object_editing_inference/` and is
available through the `object-editing` Docker Compose profile. It selects SAM 3
on an exposed CUDA GPU and YOLO11n-seg on CPU when `MODEL_BACKEND=auto`.
Configure `OBJECT_EDITING_INFERENCE_URL` to point at that service.

Optional integration settings are:

- `OBJECT_EDITING_INFERENCE_API_KEY` for the shared internal service key.
- `OBJECT_EDITING_WEBHOOK_URL` for Laravel completion callbacks.
- `OBJECT_EDITING_WEBHOOK_API_KEY` for the callback `Api-Key` header.
- `OBJECT_EDITING_REQUEST_TIMEOUT` for inference calls; it defaults to 300 seconds.

## Analyze

`POST /v1/analyze` receives the source image and Atlentis identifiers. It must
return:

```json
{
  "source_image_hash": "sha256-value",
  "model_version": "detector-version",
  "regions": [
    {
      "region_id": "table-1",
      "category_code": "fixture.dining_table",
      "label": "Dining table",
      "level": "fixture",
      "parent_id": "room-1",
      "confidence": 0.97,
      "bbox": [100, 200, 700, 900],
      "overlay": {"format": "coco_rle", "counts": "...", "size": [1200, 1600]},
      "mask_path": "private/object-masks/analysis-id/table-1.png"
    }
  ]
}
```

Levels must be `structure`, `surface`, `fixture`, or `object`. Parent IDs must
refer to another region in the same response.

## Edit

`POST /v1/edit` receives the analysis regions, expanded removal IDs, blur IDs,
and blur strength. It must perform removal first, blur the remaining selected
regions, save the output, and return:

```json
{
  "result_image_url": "https://storage.example.com/result.jpg",
  "model_version": "inpainting-version"
}
```

The service should verify `source_image_hash` before applying stored masks.
