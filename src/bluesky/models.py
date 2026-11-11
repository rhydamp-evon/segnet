from django.contrib import admin
from django.db import models
from core.models import SortableModel, TimestampModel
from django.utils.translation import gettext as _
from django.utils.safestring import mark_safe
from django.contrib import admin
from django.core.validators import MinValueValidator, MaxValueValidator


def get_sky_image_fullsize_path(instance, filename):
    return f'sky-configs/{instance.title}/full_{filename}'


def get_sky_image_fullsize_vertical_path(instance, filename):
    return f'sky-configs/{instance.title}/full_vertical_{filename}'


def get_sky_image_2k_path(instance, filename):
    return f'sky-configs/{instance.title}/2k_{filename}'


def get_sky_image_thumbnail_path(instance, filename):
    return f'sky-configs/{instance.title}/thumbnail_{filename}'


def get_sky_image_small_thumbnail_path(instance, filename):
    return f'sky-configs/{instance.title}/small_thumbnail_{filename}'


class SkyConfigStatus(models.IntegerChoices):
    UNPUBLISHED = 1, _('Unpublished')
    PUBLISHED = 2, _('Published')


def get_default_meta_data():
    return {}


class SkyConfig(TimestampModel, SortableModel):
    title = models.CharField(max_length=100, unique=True,
                             help_text="Title of sky configuration")
    sky_image_fullsize_path = models.ImageField(
        upload_to=get_sky_image_fullsize_path, max_length=256, help_text="Sky image full size path.",
        width_field='sky_image_fullsize_width', height_field='sky_image_fullsize_height')
    sky_image_fullsize_width = models.IntegerField(editable=False)
    sky_image_fullsize_height = models.IntegerField(editable=False)
    sky_image_fullsize_vertical_width = models.IntegerField(editable=False)
    sky_image_fullsize_vertical_height = models.IntegerField(editable=False)
    sky_image_fullsize_vertical_path = models.ImageField(
        upload_to=get_sky_image_fullsize_vertical_path, max_length=256, help_text="Sky image full size path.",
        width_field='sky_image_fullsize_vertical_width', height_field='sky_image_fullsize_vertical_height')

    sky_image_2k_path = models.ImageField(
        upload_to=get_sky_image_2k_path, max_length=256, help_text="Sky image 2k path.",
        width_field='sky_image_2k_width', height_field='sky_image_2k_height')
    sky_image_2k_width = models.IntegerField(editable=False)
    sky_image_2k_height = models.IntegerField(editable=False)
    sky_image_small_thumbnail_path = models.ImageField(
        upload_to=get_sky_image_small_thumbnail_path, max_length=256, help_text="Sky image small thumbnail path.")
    sky_image_thumbnail_path = models.ImageField(
        upload_to=get_sky_image_thumbnail_path, max_length=256, help_text="Sky image thumbnail path.",
        width_field='sky_image_thumbnail_width', height_field='sky_image_thumbnail_height')
    sky_image_thumbnail_width = models.IntegerField(editable=False)
    sky_image_thumbnail_height = models.IntegerField(editable=False)
    tint_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=100,
        help_text="Percentage of tint to apply on final image.")
    mod_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=50,
        help_text="Percentage of modified value to retain in final image.")
    order = models.PositiveSmallIntegerField(
        default=0, blank=False, null=False, help_text="Order in which sky config should be aligned.")
    status = models.IntegerField(
        choices=SkyConfigStatus.choices, default=SkyConfigStatus.PUBLISHED, help_text="status of configuration.")

    class Meta:
        db_table = 'tbl_sky_configs'
        ordering = ('order',)

    def __str__(self):
        return self.title

    def get_ordering_queryset(self):
        return self.all()

    @admin.display
    def sky_image_fullsize(self):
        return mark_safe(f'<img src="{self.sky_image_fullsize_path.url}" width=200px />')

    @admin.display
    def sky_image_2k(self):
        return mark_safe(f'<img src="{self.sky_image_2k_path.url}" width=200px />')

    @admin.display
    def sky_image_thumbnail(self):
        return mark_safe(f'<img src="{self.sky_image_thumbnail_path.url}" width=200px />')

    @admin.display
    def sky_thumbnail_in_list(self):
        return mark_safe(f'<img src="{self.sky_image_thumbnail_path.url}" style="width: 45px; height:45px;" />')

    sky_thumbnail_in_list.short_description = 'Thumbnail'
    sky_image_fullsize.short_description = 'Sky Image Fullsize'
    sky_image_2k.short_description = 'Sky Image 2K'
    sky_image_thumbnail.short_description = 'Sky Image Thumbnail'


def get_original_image_path(instance, filename):
    name = f'P{instance.platform_id}-C{instance.content_id}'.upper()
    return f'bluesky/{name}/{name}-{filename.upper()}'


def get_prediction_mask_path(instance, filename):
    name = f'P{instance.platform_id}-C{instance.content_id}'.upper()
    return f'bluesky/{name}/{name}-{filename}'


def get_thumbnail_relative_path_processed(instance, filename):
    name = f'P{instance.blue_sky_request.platform_id}-C{instance.blue_sky_request.content_id}-V{instance.sky_config_id}'.upper()
    return f'bluesky/{name}-{instance.blue_sky_request.content_id}/{instance.sky_config_id}/{name}-{filename}'


def get_composite_image_path(instance, filename):
    name = f'P{instance.platform_id}-C{instance.content_id}'.upper()
    return f'bluesky/{name}/{name}-{filename}'


def get_processed_image_path(instance, filename):
    name = f'P{instance.platform_id}-C{instance.content_id}-V{instance.sky_config_id}'.upper()
    return f'bluesky/{name}/{name}-{filename}'


class BlueSkyRequest(TimestampModel):
    platform_id = models.CharField(
        max_length=256, help_text="Platform ID of the user.")
    content_id = models.CharField(
        max_length=256, help_text="Content ID of the user.")
    project_id = models.CharField(max_length=256)
    content_type = models.CharField(
        max_length=10, help_text="Content type of the user.")
    original_image_path = models.URLField(
        max_length=500, help_text="Original image path.")
    is_image_downloaded = models.BooleanField(default=False, editable=False)
    prediction_mask_path = models.ImageField(
        upload_to=get_prediction_mask_path, max_length=256,
        help_text="Prediction mask generated by model.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'tbl_bluesky_requests'

    def __str__(self):
        return f'Bluesky Request ID-{self.id}'


class BlueSkyVariant(TimestampModel):
    blue_sky_request = models.ForeignKey(
        BlueSkyRequest, on_delete=models.CASCADE)
    sky_config = models.ForeignKey(
        SkyConfig, on_delete=models.CASCADE,
        help_text="Sky config which was used to create variant.")
    relative_path_processed = models.ImageField(
        upload_to=get_thumbnail_relative_path_processed, max_length=256,
        help_text="Path of processed variant.")
    tint_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=100,
        help_text="Percentage of tint to apply on final image.")
    mod_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=50,
        help_text="Percentage of modified value to retain in final image.")
    is_processed = models.BooleanField(default=False)

    class Meta:
        db_table = 'tbl_bluesky_variants'


class BlueSkyProcessed(TimestampModel):
    platform_id = models.CharField(
        max_length=256, help_text="Platform ID of the user.", editable=False)
    content_id = models.CharField(
        max_length=256, help_text="Content ID of the user.")
    platform_id = models.CharField(max_length=256, editable=False)
    project_id = models.CharField(max_length=256, editable=False)
    content_type = models.CharField(
        max_length=10, help_text="Content type of the user.", editable=False)
    blue_sky_request = models.ForeignKey(
        BlueSkyRequest, on_delete=models.SET_NULL, null=True,
        help_text="Blue sky request of the user."
    )
    original_image_path = models.URLField(
        max_length=500, help_text="Original Image which is processed.", editable=False)
    sky_config = models.ForeignKey(
        SkyConfig, on_delete=models.CASCADE, help_text="Sky Config used to process")
    relative_path_processed = models.ImageField(
        upload_to=get_processed_image_path, max_length=256, help_text="Path of processed image.")
    prediction_mask_path = models.ImageField(
        upload_to=get_prediction_mask_path, max_length=256, help_text="Prediction mask generated by model.")
    composite_image_path = models.ImageField(
        upload_to=get_composite_image_path, max_length=256, help_text="Composite mask generated by FBA model.")
    tint_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=100,
        help_text="Percentage of tint to apply on final image.")
    mod_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=50,
        help_text="Percentage of modified value to retain in final image.")
    is_processed = models.BooleanField(default=False, editable=False)
    is_pushed_to_api = models.BooleanField(
        default=False, help_text="Pushed to further PIA API.", editable=False)

    class Meta:
        db_table = 'tbl_bluesky_processed'
        verbose_name_plural = 'Bluesky Processed Images'

    def __str__(self):
        return f'Bluesky Process ID: {self.id}'

class FailedProcessStatus(models.IntegerChoices):
    NEW = 1, _('New')
    DEVELOPMENT = 2, _('Development')
    PENDING = 3, _('Pending')
    FIXED = 4, _('Fixed')
    CLOSED = 5, _('Closed')


class BlueSkyProcessFailedLog(models.Model):
    blue_sky_processed = models.ForeignKey(
        BlueSkyProcessed, on_delete=models.CASCADE)
    error = models.CharField(max_length=300)
    stack_trace = models.TextField()
    status = models.IntegerField(
        choices=FailedProcessStatus.choices, default=FailedProcessStatus.NEW, help_text="status of failed process.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'tbl_bluesky_processe_failed_log'
        verbose_name_plural = 'Bluesky Processed Failed logs'

    def __str__(self):
        return f'Bluesky Process ID - {self.blue_sky_processed_id}'
