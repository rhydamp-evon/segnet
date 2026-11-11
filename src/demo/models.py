from django.db import models
from django.contrib import admin
from django.utils.safestring import mark_safe
from django.core.validators import MinValueValidator, MaxValueValidator
from bluesky.models import SkyConfig
from core.models import TimestampModel


def get_demo_riginal_image_path(instance, filename):
    return f'blusky-demo/DEMO-{instance.id}/{filename}'

def get_rendered_image_path(instance, filename):
    return f'blusky-demo/DEMO-{instance.id}/final_{filename}'

def get_mask_image_path(instance, filename):
    return f'blusky-demo/DEMO-{instance.id}/mask_{filename}'


class BlueSkyDemo(TimestampModel):
    original_image_path = models.ImageField(
        upload_to=get_demo_riginal_image_path, max_length=256, help_text="Sky image full size path.")
    relative_path_processed = models.ImageField(
        upload_to=get_rendered_image_path, max_length=256, help_text="Sky image full size path.",)
    prediction_mask_path = models.ImageField(
        upload_to=get_mask_image_path, max_length=256, help_text="Sky image full size path.",)
    sky_config = models.ForeignKey(SkyConfig, on_delete=models.CASCADE)
    tint_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=100,
        help_text="Percentage of tint to apply on final image.")
    mod_val = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)], default=50,
        help_text="Percentage of modified value to retain in final image.")

    class Meta:
        db_table = 'tbl_blue_sky_demo'

    @property
    def title(self):
        return f'Image-{self.id}'

    @admin.display
    def original_image_preview(self):
        return mark_safe(f'<img src="{self.original_image_path.url}" height=400px />')

    @admin.display
    def processed_image_preview(self):
        return mark_safe(f'<img src="{self.relative_path_processed.url}" height=400px />')

    @admin.display
    def image_thumbnail(self):
        return mark_safe(f'<img src="{self.original_image_path.url}" style="width: 50px; height:50px;" />')


    def __str__(self):
        return f'Image-{self.id}'