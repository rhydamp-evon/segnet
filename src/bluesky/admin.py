from django.contrib import admin
from adminsortable2.admin import SortableAdminMixin
from bluesky.models import (
    SkyConfig,
    BlueSkyRequest,
    BlueSkyVariant,
    BlueSkyProcessed,
    BlueSkyProcessFailedLog,
)
from bluesky.services import create_bluesky_request, create_bluesky_variant
from bluesky.tasks import task_process_image


class SkyConfigAdmin(SortableAdminMixin, admin.ModelAdmin):
    """
    Customized Admin Page for Sky Config list view.
    """

    list_display = (
        "id",
        "order",
        "title",
        "sky_thumbnail_in_list",
        "status",
        "created_at",
        "updated_at",
    )
    list_display_links = ("id", "title", "sky_thumbnail_in_list")
    fields = (
        "title",
        "sky_image_fullsize_path",
        "sky_image_fullsize_vertical_path",
        "sky_image_fullsize",
        "sky_image_2k_path",
        "sky_image_2k",
        "sky_image_thumbnail_path",
        "sky_image_small_thumbnail_path",
        "sky_image_thumbnail",
        "order",
        "status",
        "created_at",
        "updated_at",
        "tint_val",
        "mod_val",
    )
    readonly_fields = (
        "sky_image_fullsize",
        "sky_image_2k",
        "sky_image_thumbnail",
        "order",
        "created_at",
        "updated_at",
    )
    list_filter = ("status", "created_at", "updated_at")
    search_fields = ("title",)


class BlueSkyRequestAdmin(admin.ModelAdmin):
    """
    Customized admin page for blue sky requests list view.
    """

    list_display = (
        "id",
        "platform_id",
        "content_id",
        "project_id",
        "original_image_path",
        "prediction_mask_path",
    )
    readonly_fields = ("prediction_mask_path", "is_image_downloaded")
    list_display_links = ("id", "original_image_path", "prediction_mask_path")
    list_filter = ("created_at", "updated_at")
    search_fields = ("platform_id", "platform_id", "project_id")

    def save_model(self, request, obj, form, change):
        data = {
            "platform_id": obj.platform_id,
            "content_id": obj.content_id,
            "project_id": obj.project_id,
            "content_type": obj.content_type,
            "original_image_path": obj.original_image_path,
        }
        obj, _ = create_bluesky_request(data)
        return obj


class BlueSkyVariantAdmin(admin.ModelAdmin):
    """
    Customized admin page for blue sky variant requests list view.
    """

    list_display = ("id", "blue_sky_request", "sky_config", "relative_path_processed")
    list_display_links = (
        "id",
        "relative_path_processed",
    )
    readonly_fields = ("relative_path_processed", "is_processed")
    list_filter = ("created_at", "updated_at", "is_processed")

    def save_model(self, request, obj, form, change):
        data = {
            "blue_sky_request_id": obj.blue_sky_request_id,
            "sky_config_id": obj.sky_config_id,
        }
        obj = create_bluesky_variant(data)
        return obj


class BlueSkyProcessedAdmin(admin.ModelAdmin):
    """
    Customized admin page for blue sky variant requests list view.
    """

    list_display = (
        "id",
        "content_id",
        "platform_id",
        "project_id",
        "sky_config_id",
        "is_processed",
        "is_pushed_to_api",
    )
    list_display_links = ("id", "content_id")
    list_filter = ("created_at", "updated_at", "is_processed", "is_pushed_to_api")
    search_fields = ("content_id", "platform_id", "project_id")
    readonly_fields = (
        "relative_path_processed",
        "prediction_mask_path",
        "composite_image_path",
    )

    def save_model(self, request, obj, form, change):
        data = {
            "blue_sky_request_id": obj.blue_sky_request_id,
            "sky_config_id": obj.sky_config_id,
            "content_id": obj.content_id,
        }
        task_process_image.delay(data)
        return obj


class BlueSkyProcessedFailedLogAdmin(admin.ModelAdmin):
    """
    Customized admin page for blue sky variant requests list view.
    """

    list_display = ("id", "blue_sky_processed", "error", "status", "created_at")
    list_display_links = ("id", "blue_sky_processed", "error")
    list_filter = ("created_at", "status")
    search_fields = ("error",)
    readonly_fields = ("error", "blue_sky_processed", "created_at", "stack_trace")


# Register the model with admin panel.
admin.site.register(BlueSkyRequest, BlueSkyRequestAdmin)
admin.site.register(BlueSkyVariant, BlueSkyVariantAdmin)
admin.site.register(BlueSkyProcessed, BlueSkyProcessedAdmin)
admin.site.register(SkyConfig, SkyConfigAdmin)
admin.site.register(BlueSkyProcessFailedLog, BlueSkyProcessedFailedLogAdmin)
