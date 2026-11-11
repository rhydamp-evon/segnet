import cv2
from django.contrib import admin
from django.core.files.base import ContentFile
from core.utils import BlueSkyProcessor, read_image_from_object
from demo.models import BlueSkyDemo


class BlueSkyDemoAdmin(admin.ModelAdmin):
    '''
        Customized admin page for blue sky variant requests list view.
    '''
    list_display = ('id', 'title', 'image_thumbnail',
                    'sky_config', 'relative_path_processed')
    list_display_links = ('id', 'title')
    list_filter = ('created_at', 'updated_at', 'sky_config')
    readonly_fields = ('prediction_mask_path',
                       'processed_image_preview', 'original_image_preview')
    fields = ('original_image_path', 'original_image_preview', 'sky_config', 'processed_image_preview',
              'prediction_mask_path', 'tint_val', 'mod_val')

    def save_model(self, request, obj, form, change):
        obj.save()
        img = read_image_from_object(obj.original_image_path)
        processor = BlueSkyProcessor(image=img)
        output, obj = processor.create_thumnail_image(
            obj, obj.sky_config, obj.tint_val, obj.mod_val)
        _, buf = cv2.imencode('.jpg', output)  # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        obj.relative_path_processed.save('test.jpg', content)
        return obj


admin.site.register(BlueSkyDemo, BlueSkyDemoAdmin)
