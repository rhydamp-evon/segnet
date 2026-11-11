from django.core.exceptions import ValidationError
from rest_framework import serializers
from bluesky.models import BlueSkyProcessed, SkyConfig, BlueSkyRequest, BlueSkyVariant


class SkyConfigSerializer(serializers.ModelSerializer):
    '''
    Serializes SkyConfig model. Used for GET /sky-configs/ list
    '''
    sky_image_thumbnail_path = serializers.SerializerMethodField()

    class Meta:
        model = SkyConfig
        fields = [
            'id',
            'title',
            'sky_image_thumbnail_path',
        ]

    def get_sky_image_thumbnail_path(self, instance):
        if instance.sky_image_small_thumbnail_path != '':
            return instance.sky_image_small_thumbnail_path.url
        return instance.sky_image_thumbnail_path.url


class BlueSkyRequestSerializer(serializers.ModelSerializer):
    '''
    Serializer used for request body of createRequest API endpoint.
    '''
    original_image_path = serializers.URLField()
    tint_val = serializers.IntegerField(required=False)
    mod_val = serializers.IntegerField(required=False)

    class Meta:
        model = BlueSkyRequest
        fields = [
            'id',
            'platform_id',
            'content_id',
            'project_id',
            'content_type',
            'original_image_path',
            'tint_val',
            'mod_val',
            'created_at',
        ]


class BlueSkyVariantSerializer(serializers.ModelSerializer):
    '''
    Serializes response for createRequest & createVariant API endpoints.
    '''
    class Meta:
        model = BlueSkyVariant
        fields = [
            'blue_sky_request_id',
            'sky_config_id',
            'relative_path_processed',
            'tint_val',
            'mod_val',
            'created_at'
        ]


class BlueSkyNewVariantSerializer(serializers.Serializer):
    '''
    Serializer used to represent format request body also validate input for 
    createVariant & processVariant API endpoints.
    '''
    blue_sky_request_id = serializers.IntegerField()
    sky_config_id = serializers.IntegerField()
    tint_val = serializers.IntegerField(required=False)
    mod_val = serializers.IntegerField(required=False)

    def validate_blue_sky_request_id(self, value):
        if not BlueSkyRequest.objects.filter(id=value).exists():
            raise ValidationError("Invalid blue_sky_request_id.")
        return value

    def validate_sky_config_id(self, value):
        if not SkyConfig.objects.filter(id=value).exists():
            raise ValidationError("Invalid sky_config_id.")
        return value


class ProcessVariantSerializer(BlueSkyNewVariantSerializer):
    content_id = serializers.CharField()


class PushToWebhookSerializer(serializers.Serializer):
    content_id = serializers.CharField()

    def validate_content_id(self, value):
        if not BlueSkyProcessed.objects.filter(content_id=value).exists():
            raise ValidationError(
                "Request with given content id does not exist.")
        return value


class LocalPreviewSerializer(serializers.Serializer):
    """Validate two locally uploaded images for the synchronous preview API."""

    image = serializers.ImageField(help_text="Photo whose sky will be replaced.")
    sky_image = serializers.ImageField(help_text="Replacement sky image.")
    mode = serializers.ChoiceField(
        choices=["preview", "matting"],
        required=False,
        default="preview",
        help_text="preview runs SkySegNet; matting runs SkySegNet and FBA Matting.",
    )
    tint_val = serializers.IntegerField(
        required=False, default=100, min_value=0, max_value=100
    )
    mod_val = serializers.IntegerField(
        required=False, default=50, min_value=0, max_value=100
    )
