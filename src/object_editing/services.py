import requests
from django.conf import settings
from rest_framework import serializers


class InferenceServiceError(Exception):
    pass


class InferenceServiceUnavailable(InferenceServiceError):
    pass


def expand_remove_selection(regions, requested_ids):
    """Return selected region IDs plus all of their descendants."""
    children_by_parent = {}
    known_ids = set()

    for region in regions:
        known_ids.add(region.region_id)
        if region.parent_id:
            children_by_parent.setdefault(region.parent.region_id, []).append(
                region.region_id
            )

    unknown = set(requested_ids) - known_ids
    if unknown:
        raise serializers.ValidationError(
            {"region_ids": f"Unknown region IDs: {', '.join(sorted(unknown))}."}
        )

    expanded = set(requested_ids)
    pending = list(requested_ids)
    while pending:
        current = pending.pop()
        for child_id in children_by_parent.get(current, []):
            if child_id not in expanded:
                expanded.add(child_id)
                pending.append(child_id)
    return sorted(expanded)


class ObjectEditingInferenceClient:
    """Client for the independently deployed object-editing model service."""

    def __init__(self):
        self.base_url = settings.OBJECT_EDITING_INFERENCE_URL.rstrip("/")
        self.timeout = settings.OBJECT_EDITING_REQUEST_TIMEOUT

    def _post(self, path, payload):
        if not self.base_url:
            raise InferenceServiceError(
                "OBJECT_EDITING_INFERENCE_URL is not configured."
            )

        headers = {}
        if settings.OBJECT_EDITING_INFERENCE_API_KEY:
            headers["Api-Key"] = settings.OBJECT_EDITING_INFERENCE_API_KEY

        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code >= 500:
            raise InferenceServiceUnavailable(
                f"Inference service is unavailable ({response.status_code})."
            )
        try:
            response.raise_for_status()
            return response.json()
        except (ValueError, requests.HTTPError) as exc:
            detail = response.text[:500]
            raise InferenceServiceError(
                f"Inference service returned an invalid response: {detail}"
            ) from exc

    def analyze(self, analysis):
        return self._post(
            "/v1/analyze",
            {
                "analysis_id": str(analysis.id),
                "source_image_url": analysis.source_image_url,
                "platform_id": analysis.platform_id,
                "project_id": analysis.project_id,
                "content_id": analysis.content_id,
                "content_type": analysis.content_type,
            },
        )

    def edit(self, job, regions):
        return self._post(
            "/v1/edit",
            {
                "job_id": str(job.id),
                "analysis_id": str(job.analysis_id),
                "source_image_url": job.analysis.source_image_url,
                "source_image_hash": job.analysis.source_image_hash,
                "remove_region_ids": job.expanded_remove_ids,
                "blur_region_ids": job.requested_blur_ids,
                "blur_strength": job.blur_strength,
                "regions": [
                    {
                        "region_id": region.region_id,
                        "category_code": region.category_code,
                        "level": region.level,
                        "parent_id": (
                            region.parent.region_id if region.parent_id else None
                        ),
                        "bbox": region.bbox,
                        "mask_path": region.mask_path,
                    }
                    for region in regions
                ],
            },
        )
