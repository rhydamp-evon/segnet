import logging

import requests
from celery import shared_task
from django.conf import settings
from django.db import transaction

from object_editing.models import (
    AnalysisStatus,
    DetectedRegion,
    EditJobStatus,
    ObjectAnalysis,
    ObjectEditJob,
    RegionLevel,
)
from object_editing.services import (
    InferenceServiceError,
    InferenceServiceUnavailable,
    ObjectEditingInferenceClient,
)

logger = logging.getLogger("error_logger")


def _retry_network_error(task, exc):
    if task.request.retries >= task.max_retries:
        return False
    raise task.retry(exc=exc, countdown=2 ** (task.request.retries + 1))


def _store_regions(analysis, payload):
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise InferenceServiceError("Analysis response must contain a regions list.")

    valid_levels = {choice for choice, _ in RegionLevel.choices}
    level_order = {
        RegionLevel.STRUCTURE: 0,
        RegionLevel.SURFACE: 1,
        RegionLevel.FIXTURE: 2,
        RegionLevel.OBJECT: 3,
    }
    seen_ids = set()
    with transaction.atomic():
        analysis.regions.all().delete()
        region_objects = {}
        parent_ids = {}

        for item in regions:
            region_id = str(item.get("region_id", "")).strip()
            level = item.get("level")
            category_code = item.get("category_code") or "uncategorized"
            if not region_id or region_id in seen_ids:
                raise InferenceServiceError(
                    "Every detected region must have a unique region_id."
                )
            if level not in valid_levels:
                raise InferenceServiceError(
                    f"Region {region_id} has an invalid hierarchy level."
                )
            seen_ids.add(region_id)
            region = DetectedRegion.objects.create(
                analysis=analysis,
                region_id=region_id,
                category_code=category_code,
                label=item.get("label") or category_code,
                level=level,
                confidence=item.get("confidence"),
                bbox=item.get("bbox") or [],
                overlay=item.get("overlay") or {},
                mask_path=item.get("mask_path") or "",
            )
            region_objects[region_id] = region
            parent_ids[region_id] = item.get("parent_id")

        for region_id, parent_region_id in parent_ids.items():
            if not parent_region_id:
                continue
            parent = region_objects.get(str(parent_region_id))
            if not parent:
                raise InferenceServiceError(
                    f"Region {region_id} refers to an unknown parent."
                )
            region = region_objects[region_id]
            if level_order[parent.level] >= level_order[region.level]:
                raise InferenceServiceError(
                    f"Region {region_id} has an invalid parent hierarchy."
                )
            region.parent = parent
            region.save(update_fields=("parent", "updated_at"))


@shared_task(bind=True, max_retries=3)
def task_analyze_object_image(self, analysis_id):
    analysis = ObjectAnalysis.objects.get(id=analysis_id)
    if analysis.status == AnalysisStatus.READY:
        return True

    analysis.status = AnalysisStatus.PROCESSING
    analysis.error = ""
    analysis.save(update_fields=("status", "error", "updated_at"))
    try:
        payload = ObjectEditingInferenceClient().analyze(analysis)
        source_image_hash = payload.get("source_image_hash")
        if not source_image_hash:
            raise InferenceServiceError(
                "Analysis response must contain source_image_hash."
            )
        _store_regions(analysis, payload)
        analysis.source_image_hash = source_image_hash
        analysis.model_version = payload.get("model_version", "")
        analysis.status = AnalysisStatus.READY
        analysis.save(
            update_fields=(
                "source_image_hash",
                "model_version",
                "status",
                "updated_at",
            )
        )
        return True
    except (
        requests.Timeout,
        requests.ConnectionError,
        InferenceServiceUnavailable,
    ) as exc:
        if _retry_network_error(self, exc) is False:
            analysis.status = AnalysisStatus.FAILED
            analysis.error = str(exc)
            analysis.save(update_fields=("status", "error", "updated_at"))
        return False
    except Exception as exc:
        logger.exception(exc)
        analysis.status = AnalysisStatus.FAILED
        analysis.error = str(exc)
        analysis.save(update_fields=("status", "error", "updated_at"))
        return False


@shared_task(bind=True, max_retries=3)
def task_process_object_edit(self, job_id):
    job = ObjectEditJob.objects.select_related("analysis").get(id=job_id)
    if job.status == EditJobStatus.COMPLETED:
        return True

    job.status = EditJobStatus.PROCESSING
    job.error = ""
    job.save(update_fields=("status", "error", "updated_at"))
    try:
        regions = list(job.analysis.regions.select_related("parent").all())
        payload = ObjectEditingInferenceClient().edit(job, regions)
        result_image_url = payload.get("result_image_url")
        if not result_image_url:
            raise InferenceServiceError(
                "Edit response must contain result_image_url."
            )
        job.result_image_url = result_image_url
        job.model_version = payload.get("model_version", "")
        job.status = EditJobStatus.COMPLETED
        job.save(
            update_fields=(
                "result_image_url",
                "model_version",
                "status",
                "updated_at",
            )
        )
        task_deliver_object_edit_webhook.delay(str(job.id))
        return True
    except (
        requests.Timeout,
        requests.ConnectionError,
        InferenceServiceUnavailable,
    ) as exc:
        if _retry_network_error(self, exc) is False:
            job.status = EditJobStatus.FAILED
            job.error = str(exc)
            job.save(update_fields=("status", "error", "updated_at"))
            task_deliver_object_edit_webhook.delay(str(job.id))
        return False
    except Exception as exc:
        logger.exception(exc)
        job.status = EditJobStatus.FAILED
        job.error = str(exc)
        job.save(update_fields=("status", "error", "updated_at"))
        task_deliver_object_edit_webhook.delay(str(job.id))
        return False


@shared_task(bind=True, max_retries=5)
def task_deliver_object_edit_webhook(self, job_id):
    if not settings.OBJECT_EDITING_WEBHOOK_URL:
        return False

    job = ObjectEditJob.objects.select_related("analysis").get(id=job_id)
    if job.webhook_delivered:
        return True

    payload = {
        "event": f"object_edit.{job.status}",
        "job_id": str(job.id),
        "analysis_id": str(job.analysis_id),
        "platform_id": job.analysis.platform_id,
        "project_id": job.analysis.project_id,
        "content_id": job.analysis.content_id,
        "content_type": job.analysis.content_type,
        "result_image_url": job.result_image_url or None,
        "error": job.error or None,
    }
    headers = {"Content-Type": "application/json"}
    if settings.OBJECT_EDITING_WEBHOOK_API_KEY:
        headers["Api-Key"] = settings.OBJECT_EDITING_WEBHOOK_API_KEY

    try:
        response = requests.post(
            settings.OBJECT_EDITING_WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1))

    job.webhook_delivered = True
    job.save(update_fields=("webhook_delivered", "updated_at"))
    return True
