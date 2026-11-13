from rest_framework import serializers

from object_editing.models import (
    AnalysisStatus,
    DetectedRegion,
    ObjectAnalysis,
    ObjectEditJob,
)
from object_editing.services import expand_remove_selection


class ObjectAnalysisCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ObjectAnalysis
        fields = (
            "id",
            "platform_id",
            "content_id",
            "project_id",
            "content_type",
            "source_image_url",
            "status",
            "created_at",
        )
        read_only_fields = ("id", "status", "created_at")


class DetectedRegionSerializer(serializers.ModelSerializer):
    parent_id = serializers.SerializerMethodField()

    class Meta:
        model = DetectedRegion
        fields = (
            "region_id",
            "category_code",
            "label",
            "level",
            "parent_id",
            "confidence",
            "bbox",
            "overlay",
        )

    def get_parent_id(self, instance):
        return instance.parent.region_id if instance.parent_id else None


class ObjectAnalysisDetailSerializer(serializers.ModelSerializer):
    regions = DetectedRegionSerializer(many=True, read_only=True)

    class Meta:
        model = ObjectAnalysis
        fields = (
            "id",
            "platform_id",
            "content_id",
            "project_id",
            "content_type",
            "source_image_url",
            "source_image_hash",
            "status",
            "model_version",
            "error",
            "regions",
            "created_at",
            "updated_at",
        )


class ObjectEditJobCreateSerializer(serializers.Serializer):
    analysis_id = serializers.UUIDField()
    remove_region_ids = serializers.ListField(
        child=serializers.CharField(max_length=128), required=False, default=list
    )
    blur_region_ids = serializers.ListField(
        child=serializers.CharField(max_length=128), required=False, default=list
    )
    blur_strength = serializers.IntegerField(
        required=False, default=25, min_value=1, max_value=100
    )
    idempotency_key = serializers.CharField(max_length=255)

    def validate(self, attrs):
        try:
            analysis = ObjectAnalysis.objects.get(id=attrs["analysis_id"])
        except ObjectAnalysis.DoesNotExist as exc:
            raise serializers.ValidationError(
                {"analysis_id": "Analysis does not exist."}
            ) from exc

        if analysis.status != AnalysisStatus.READY:
            raise serializers.ValidationError(
                {"analysis_id": "Analysis is not ready for editing."}
            )

        remove_ids = list(dict.fromkeys(attrs["remove_region_ids"]))
        blur_ids = list(dict.fromkeys(attrs["blur_region_ids"]))
        if not remove_ids and not blur_ids:
            raise serializers.ValidationError(
                "At least one remove or blur region must be selected."
            )

        regions = list(analysis.regions.select_related("parent").all())
        requested_ids = set(remove_ids).union(blur_ids)
        non_editable_ids = sorted(
            region.region_id
            for region in regions
            if region.region_id in requested_ids and not region.mask_path
        )
        if non_editable_ids:
            raise serializers.ValidationError(
                {
                    "region_ids": (
                        "Regions do not contain editable masks: "
                        f"{', '.join(non_editable_ids)}."
                    )
                }
            )

        expanded_remove_ids = expand_remove_selection(regions, remove_ids)
        known_ids = {region.region_id for region in regions}
        unknown_blur_ids = set(blur_ids) - known_ids
        if unknown_blur_ids:
            raise serializers.ValidationError(
                {
                    "blur_region_ids": (
                        "Unknown region IDs: "
                        f"{', '.join(sorted(unknown_blur_ids))}."
                    )
                }
            )

        conflicts = set(expanded_remove_ids).intersection(blur_ids)
        if conflicts:
            raise serializers.ValidationError(
                {
                    "blur_region_ids": (
                        "Removed regions cannot also be blurred: "
                        f"{', '.join(sorted(conflicts))}."
                    )
                }
            )

        attrs["analysis"] = analysis
        attrs["remove_region_ids"] = remove_ids
        attrs["blur_region_ids"] = blur_ids
        attrs["expanded_remove_ids"] = expanded_remove_ids
        return attrs

    def create(self, validated_data):
        validated_data.pop("analysis_id")
        job, created = ObjectEditJob.objects.get_or_create(
            idempotency_key=validated_data["idempotency_key"],
            defaults={
                "analysis": validated_data["analysis"],
                "requested_remove_ids": validated_data["remove_region_ids"],
                "requested_blur_ids": validated_data["blur_region_ids"],
                "expanded_remove_ids": validated_data["expanded_remove_ids"],
                "blur_strength": validated_data["blur_strength"],
            },
        )
        if not created:
            expected = (
                job.analysis_id == validated_data["analysis"].id
                and job.requested_remove_ids
                == validated_data["remove_region_ids"]
                and job.requested_blur_ids == validated_data["blur_region_ids"]
                and job.blur_strength == validated_data["blur_strength"]
            )
            if not expected:
                raise serializers.ValidationError(
                    {"idempotency_key": "Key was already used for another request."}
                )
            self.reused_existing = True
            return job

        self.reused_existing = False
        return job


class ObjectEditJobDetailSerializer(serializers.ModelSerializer):
    analysis_id = serializers.UUIDField(read_only=True)

    class Meta:
        model = ObjectEditJob
        fields = (
            "id",
            "analysis_id",
            "idempotency_key",
            "requested_remove_ids",
            "requested_blur_ids",
            "expanded_remove_ids",
            "blur_strength",
            "status",
            "result_image_url",
            "model_version",
            "error",
            "created_at",
            "updated_at",
        )
