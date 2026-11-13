# Generated for the independent object editing application.
import uuid

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ObjectAnalysis",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("platform_id", models.CharField(max_length=256)),
                ("content_id", models.CharField(max_length=256)),
                ("project_id", models.CharField(max_length=256)),
                ("content_type", models.CharField(max_length=32)),
                ("source_image_url", models.URLField(max_length=1000)),
                ("source_image_hash", models.CharField(blank=True, max_length=128)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("queued", "Queued"),
                            ("processing", "Processing"),
                            ("ready", "Ready"),
                            ("failed", "Failed"),
                        ],
                        default="queued",
                        max_length=16,
                    ),
                ),
                ("model_version", models.CharField(blank=True, max_length=128)),
                ("error", models.TextField(blank=True)),
            ],
            options={
                "db_table": "tbl_object_analyses",
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="ObjectEditJob",
            fields=[
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("idempotency_key", models.CharField(max_length=255, unique=True)),
                ("requested_remove_ids", models.JSONField(default=list)),
                ("requested_blur_ids", models.JSONField(default=list)),
                ("expanded_remove_ids", models.JSONField(default=list)),
                (
                    "blur_strength",
                    models.PositiveSmallIntegerField(
                        default=25,
                        validators=[
                            django.core.validators.MinValueValidator(1),
                            django.core.validators.MaxValueValidator(100),
                        ],
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("queued", "Queued"),
                            ("processing", "Processing"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                        ],
                        default="queued",
                        max_length=16,
                    ),
                ),
                ("result_image_url", models.URLField(blank=True, max_length=1000)),
                ("model_version", models.CharField(blank=True, max_length=128)),
                ("error", models.TextField(blank=True)),
                ("webhook_delivered", models.BooleanField(default=False)),
                (
                    "analysis",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="edit_jobs",
                        to="object_editing.objectanalysis",
                    ),
                ),
            ],
            options={
                "db_table": "tbl_object_edit_jobs",
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="DetectedRegion",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("region_id", models.CharField(max_length=128)),
                ("category_code", models.CharField(max_length=128)),
                ("label", models.CharField(max_length=255)),
                (
                    "level",
                    models.CharField(
                        choices=[
                            ("structure", "Structure"),
                            ("surface", "Surface"),
                            ("fixture", "Fixture"),
                            ("object", "Object"),
                        ],
                        max_length=16,
                    ),
                ),
                ("confidence", models.FloatField(blank=True, null=True)),
                ("bbox", models.JSONField(blank=True, default=list)),
                ("overlay", models.JSONField(blank=True, default=dict)),
                ("mask_path", models.CharField(blank=True, max_length=1000)),
                (
                    "analysis",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="regions",
                        to="object_editing.objectanalysis",
                    ),
                ),
                (
                    "parent",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="children",
                        to="object_editing.detectedregion",
                    ),
                ),
            ],
            options={
                "db_table": "tbl_detected_regions",
                "ordering": ("created_at", "id"),
            },
        ),
        migrations.AddConstraint(
            model_name="detectedregion",
            constraint=models.UniqueConstraint(
                fields=("analysis", "region_id"),
                name="unique_region_per_object_analysis",
            ),
        ),
    ]
