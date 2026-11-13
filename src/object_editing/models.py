import uuid

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.translation import gettext_lazy as _

from core.models import TimestampModel


class AnalysisStatus(models.TextChoices):
    QUEUED = "queued", _("Queued")
    PROCESSING = "processing", _("Processing")
    READY = "ready", _("Ready")
    FAILED = "failed", _("Failed")


class EditJobStatus(models.TextChoices):
    QUEUED = "queued", _("Queued")
    PROCESSING = "processing", _("Processing")
    COMPLETED = "completed", _("Completed")
    FAILED = "failed", _("Failed")


class RegionLevel(models.TextChoices):
    STRUCTURE = "structure", _("Structure")
    SURFACE = "surface", _("Surface")
    FIXTURE = "fixture", _("Fixture")
    OBJECT = "object", _("Object")


class ObjectAnalysis(TimestampModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    platform_id = models.CharField(max_length=256)
    content_id = models.CharField(max_length=256)
    project_id = models.CharField(max_length=256)
    content_type = models.CharField(max_length=32)
    source_image_url = models.URLField(max_length=1000)
    source_image_hash = models.CharField(max_length=128, blank=True)
    status = models.CharField(
        max_length=16,
        choices=AnalysisStatus.choices,
        default=AnalysisStatus.QUEUED,
    )
    model_version = models.CharField(max_length=128, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        db_table = "tbl_object_analyses"
        ordering = ("-created_at",)

    def __str__(self):
        return f"Object analysis {self.id}"


class DetectedRegion(TimestampModel):
    analysis = models.ForeignKey(
        ObjectAnalysis, related_name="regions", on_delete=models.CASCADE
    )
    region_id = models.CharField(max_length=128)
    category_code = models.CharField(max_length=128)
    label = models.CharField(max_length=255)
    level = models.CharField(max_length=16, choices=RegionLevel.choices)
    parent = models.ForeignKey(
        "self",
        related_name="children",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    confidence = models.FloatField(null=True, blank=True)
    bbox = models.JSONField(default=list, blank=True)
    overlay = models.JSONField(default=dict, blank=True)
    mask_path = models.CharField(max_length=1000, blank=True)

    class Meta:
        db_table = "tbl_detected_regions"
        ordering = ("created_at", "id")
        constraints = [
            models.UniqueConstraint(
                fields=("analysis", "region_id"),
                name="unique_region_per_object_analysis",
            )
        ]

    def __str__(self):
        return f"{self.label} ({self.region_id})"


class ObjectEditJob(TimestampModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(
        ObjectAnalysis, related_name="edit_jobs", on_delete=models.PROTECT
    )
    idempotency_key = models.CharField(max_length=255, unique=True)
    requested_remove_ids = models.JSONField(default=list)
    requested_blur_ids = models.JSONField(default=list)
    expanded_remove_ids = models.JSONField(default=list)
    blur_strength = models.PositiveSmallIntegerField(
        default=25,
        validators=(MinValueValidator(1), MaxValueValidator(100)),
    )
    status = models.CharField(
        max_length=16,
        choices=EditJobStatus.choices,
        default=EditJobStatus.QUEUED,
    )
    result_image_url = models.URLField(max_length=1000, blank=True)
    model_version = models.CharField(max_length=128, blank=True)
    error = models.TextField(blank=True)
    webhook_delivered = models.BooleanField(default=False)

    class Meta:
        db_table = "tbl_object_edit_jobs"
        ordering = ("-created_at",)

    def __str__(self):
        return f"Object edit job {self.id}"
