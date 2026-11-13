from unittest.mock import patch

from django.test import TestCase

from object_editing.models import (
    AnalysisStatus,
    DetectedRegion,
    EditJobStatus,
    ObjectAnalysis,
    RegionLevel,
)
from object_editing.serializers import ObjectEditJobCreateSerializer
from object_editing.services import expand_remove_selection
from object_editing.tasks import task_analyze_object_image, task_process_object_edit


class ObjectEditingSelectionTests(TestCase):
    def setUp(self):
        self.analysis = ObjectAnalysis.objects.create(
            platform_id="platform-1",
            content_id="content-1",
            project_id="project-1",
            content_type="photo",
            source_image_url="https://images.example.com/photo.jpg",
            status=AnalysisStatus.READY,
        )
        self.room = DetectedRegion.objects.create(
            analysis=self.analysis,
            region_id="room-1",
            category_code="structure.living_room",
            label="Living room",
            level=RegionLevel.STRUCTURE,
        )
        self.table = DetectedRegion.objects.create(
            analysis=self.analysis,
            region_id="table-1",
            category_code="fixture.dining_table",
            label="Dining table",
            level=RegionLevel.FIXTURE,
            parent=self.room,
            mask_path="object-editing/test/table.png",
        )
        self.plate = DetectedRegion.objects.create(
            analysis=self.analysis,
            region_id="plate-1",
            category_code="object.plate",
            label="Plate",
            level=RegionLevel.OBJECT,
            parent=self.table,
            mask_path="object-editing/test/plate.png",
        )

    def test_parent_removal_expands_to_descendants(self):
        regions = list(self.analysis.regions.select_related("parent"))
        expanded = expand_remove_selection(regions, ["table-1"])
        self.assertEqual(expanded, ["plate-1", "table-1"])

    def test_removed_descendant_cannot_be_blurred(self):
        serializer = ObjectEditJobCreateSerializer(
            data={
                "analysis_id": str(self.analysis.id),
                "remove_region_ids": ["table-1"],
                "blur_region_ids": ["plate-1"],
                "idempotency_key": "request-1",
            }
        )
        self.assertFalse(serializer.is_valid())
        self.assertIn("blur_region_ids", serializer.errors)

    def test_empty_edit_is_rejected(self):
        serializer = ObjectEditJobCreateSerializer(
            data={
                "analysis_id": str(self.analysis.id),
                "idempotency_key": "request-2",
            }
        )
        self.assertFalse(serializer.is_valid())

    def test_region_without_mask_is_rejected(self):
        serializer = ObjectEditJobCreateSerializer(
            data={
                "analysis_id": str(self.analysis.id),
                "remove_region_ids": ["room-1"],
                "idempotency_key": "request-without-mask",
            }
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("region_ids", serializer.errors)

    def test_idempotent_request_returns_existing_job(self):
        data = {
            "analysis_id": str(self.analysis.id),
            "remove_region_ids": ["plate-1"],
            "idempotency_key": "request-3",
        }
        first = ObjectEditJobCreateSerializer(data=data)
        self.assertTrue(first.is_valid(), first.errors)
        first_job = first.save()

        second = ObjectEditJobCreateSerializer(data=data)
        self.assertTrue(second.is_valid(), second.errors)
        second_job = second.save()

        self.assertEqual(first_job.id, second_job.id)
        self.assertTrue(second.reused_existing)

    @patch("object_editing.tasks.ObjectEditingInferenceClient.analyze")
    def test_analysis_task_stores_inference_hierarchy(self, analyze):
        self.analysis.regions.all().delete()
        self.analysis.status = AnalysisStatus.QUEUED
        self.analysis.save(update_fields=("status", "updated_at"))
        analyze.return_value = {
            "source_image_hash": "sha256-test",
            "model_version": "detector-test",
            "regions": [
                {
                    "region_id": "room-new",
                    "category_code": "structure.living_room",
                    "label": "Living room",
                    "level": "structure",
                },
                {
                    "region_id": "table-new",
                    "category_code": "fixture.dining_table",
                    "label": "Dining table",
                    "level": "fixture",
                    "parent_id": "room-new",
                    "mask_path": "private/table-new.png",
                },
            ],
        }

        self.assertTrue(task_analyze_object_image.run(str(self.analysis.id)))
        self.analysis.refresh_from_db()
        table = self.analysis.regions.get(region_id="table-new")

        self.assertEqual(self.analysis.status, AnalysisStatus.READY)
        self.assertEqual(self.analysis.source_image_hash, "sha256-test")
        self.assertEqual(table.parent.region_id, "room-new")

    @patch("object_editing.tasks.task_deliver_object_edit_webhook.delay")
    @patch("object_editing.tasks.ObjectEditingInferenceClient.edit")
    def test_edit_task_stores_result(self, edit, deliver_webhook):
        serializer = ObjectEditJobCreateSerializer(
            data={
                "analysis_id": str(self.analysis.id),
                "remove_region_ids": ["plate-1"],
                "idempotency_key": "request-4",
            }
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        job = serializer.save()
        edit.return_value = {
            "result_image_url": "https://images.example.com/result.jpg",
            "model_version": "inpainting-test",
        }

        self.assertTrue(task_process_object_edit.run(str(job.id)))
        job.refresh_from_db()

        self.assertEqual(job.status, EditJobStatus.COMPLETED)
        self.assertEqual(
            job.result_image_url, "https://images.example.com/result.jpg"
        )
        deliver_webhook.assert_called_once_with(str(job.id))
