from django.contrib import admin

from object_editing.models import DetectedRegion, ObjectAnalysis, ObjectEditJob


@admin.register(ObjectAnalysis)
class ObjectAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "project_id", "content_id", "status", "created_at")
    list_filter = ("status", "content_type")
    search_fields = ("project_id", "content_id", "platform_id")


@admin.register(DetectedRegion)
class DetectedRegionAdmin(admin.ModelAdmin):
    list_display = ("region_id", "label", "level", "analysis", "parent")
    list_filter = ("level",)
    search_fields = ("region_id", "label", "category_code")


@admin.register(ObjectEditJob)
class ObjectEditJobAdmin(admin.ModelAdmin):
    list_display = ("id", "analysis", "status", "webhook_delivered", "created_at")
    list_filter = ("status", "webhook_delivered")
    search_fields = ("idempotency_key", "analysis__content_id")
