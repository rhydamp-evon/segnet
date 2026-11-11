import requests
import json
from io import BytesIO

import cv2
import numpy as np


from django.conf import settings
from django.http import FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from rest_framework import generics, serializers
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAdminUser, IsAuthenticated
from rest_framework_api_key.permissions import HasAPIKey
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.parsers import FormParser, MultiPartParser
from drf_yasg.utils import swagger_auto_schema
from sentry_sdk.api import capture_exception

from bluesky.models import BlueSkyProcessed, SkyConfig, BlueSkyRequest, BlueSkyVariant,\
    SkyConfigStatus, BlueSkyProcessFailedLog, FailedProcessStatus
from bluesky import serializers as bluesky_serializers
from bluesky.services import create_bluesky_request, create_bluesky_variant
from bluesky.tasks import task_process_image, task_reprocess_failed_images
from atlentis_ml_api.settings import BLUESKY_WEBHOOK_URL, BLUESKY_WEBHOOK_API_KEY
from core.utils import BlueSkyProcessor
from core.services.matting import get_composite_image


class SkyConfigListView(generics.ListAPIView):
    '''
    Returns a list of available sky configuration.
    '''
    queryset = SkyConfig.objects.filter(
        status=SkyConfigStatus.PUBLISHED.value).order_by('order')
    serializer_class = bluesky_serializers.SkyConfigSerializer

    @method_decorator(cache_page(60))
    def dispatch(self, *args, **kwargs):
        return super(SkyConfigListView, self).dispatch(*args, **kwargs)


class SkyConfigDetailView(generics.RetrieveAPIView):
    '''
    Returns detailed info of sky config.
    '''
    queryset = SkyConfig.objects.all()
    serializer_class = bluesky_serializers.SkyConfigSerializer


class CreateRequestAPIView(generics.CreateAPIView):
    '''
    API endpoint to create a new bluesky image,
    returns bluesky preview image with skyconfig at order 1.
    '''
    queryset = BlueSkyRequest.objects.all()
    serializer_class = bluesky_serializers.BlueSkyRequestSerializer
    resp_serializer_class = bluesky_serializers.BlueSkyVariantSerializer

    @swagger_auto_schema(responses={status.HTTP_200_OK: bluesky_serializers.BlueSkyVariantSerializer})
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        _, bluesky_variant = create_bluesky_request(serializer.validated_data)
        serializer = self.resp_serializer_class(bluesky_variant)
        return Response(serializer.data, status=status.HTTP_200_OK)


class CreateRequestVariantAPIView(generics.CreateAPIView):
    '''
    API endpoint to create a new variant of bluesky image, Returns bluesky preview image 
    with config passed in sky_config_id input parameter.
    '''
    queryset = BlueSkyVariant.objects.all()
    serializer_class = bluesky_serializers.BlueSkyNewVariantSerializer
    resp_serializer_class = bluesky_serializers.BlueSkyVariantSerializer

    @swagger_auto_schema(responses={status.HTTP_200_OK: bluesky_serializers.BlueSkyVariantSerializer})
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        bluesky_variant = create_bluesky_variant(serializer.validated_data)
        serializer = self.resp_serializer_class(bluesky_variant)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ProcessVariantAPIView(generics.CreateAPIView):
    '''
    API endpoint to process a new variant of bluesky image. Create a new async task to for further
    processing. Retuens 200 response when task is added.
    '''
    queryset = BlueSkyProcessed.objects.all()
    serializer_class = bluesky_serializers.ProcessVariantSerializer

    @swagger_auto_schema(responses={status.HTTP_200_OK: bluesky_serializers.BlueSkyVariantSerializer})
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        task_process_image.delay(serializer.validated_data)
        return Response(status=status.HTTP_200_OK)


class RepushToWebhookAPIView(generics.GenericAPIView):
    '''
    If an image is processed (ProcessVariant) but is not pushed to webhook api then
    this endpoint will again hit the webhook.
    This endpoint will be used by a cron script run on other service. 
    '''

    serializer_class = bluesky_serializers.PushToWebhookSerializer

    def post(self, request, format=None):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            content_id = serializer.data.get('content_id')
            bluesky_process = BlueSkyProcessed.objects.get(
                content_id=content_id)
            # if bluesky_process.is_pushed_to_api == True:
            #     return Response({"detail": "Already pushed to api."}, status=status.HTTP_400_BAD_REQUEST)
            if bluesky_process.is_processed == False:
                return Response({"detail": "Image not processed."}, status=status.HTTP_400_BAD_REQUEST)

            data = {
                'project_id': bluesky_process.project_id,
                'platform_id': bluesky_process.platform_id,
                'content_id': bluesky_process.content_id,
                'content_type': bluesky_process.content_type,
                's3_relative_path': bluesky_process.relative_path_processed.url,
            }

            # UNCOMMENT if the webhook is to be hit with the data.
            # headers = {
            #     'Api-Key': BLUESKY_WEBHOOK_API_KEY,
            #     'Content-Type': 'application/json'
            # }
            # try:
            #     res = requests.post(BLUESKY_WEBHOOK_URL,
            #                         data=json.dumps(data), headers=headers)

            #     if res.status_code == 200:
            #         bluesky_process.is_pushed_to_api = True
            #         bluesky_process.save(update_fields=(
            #             'is_pushed_to_api', 'updated_at'))

            # except Exception as e:
            #     capture_exception(e)

            return Response(data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ProcessFailedImagesAPIView(APIView):
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    permission_classes = [HasAPIKey | IsAdminUser]

    def post(self, request, format=None):
        try:
            failed_process = BlueSkyProcessFailedLog.objects.filter(status=FailedProcessStatus.FIXED.value) \
                .select_related('blue_sky_processed')
            for obj in failed_process:
                data = {
                    "failed_log_id": obj.id,
                    "blue_sky_request_id": obj.blue_sky_processed.blue_sky_request_id,
                    "sky_config_id": obj.blue_sky_processed.sky_config_id,
                    "content_id": obj.blue_sky_processed.content_id
                }
                task_reprocess_failed_images.delay(data)
            # data = {'message': "Fixed images are added to queue."}
            return Response({'message': "Fixed images are added to queue."}, status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            capture_exception(e)


class LocalPreviewAPIView(APIView):
    """Create a synchronous sky-replacement preview from two uploaded files.

    This endpoint exists for local development and Postman testing.  It does
    not create database records or enqueue a Celery task.  It uses SkySegNet's
    mask and the normal OpenCV blending pipeline, but skips FBA Matting so the
    response is fast enough for an HTTP request.  Pass mode=matting to run FBA
    Matting synchronously too; that is slower, but still does not use Celery.
    """

    parser_classes = [MultiPartParser, FormParser]
    serializer_class = bluesky_serializers.LocalPreviewSerializer
    permission_classes = [AllowAny] if settings.DEBUG else [HasAPIKey]

    @staticmethod
    def _decode_image(upload):
        image = cv2.imdecode(
            np.frombuffer(upload.read(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise serializers.ValidationError("Invalid image file.")
        return image

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        original = self._decode_image(data["image"])
        sky = self._decode_image(data["sky_image"])
        processor = BlueSkyProcessor(image=original)

        trimap, mask = processor.create_mask()
        if data["mode"] == "matting":
            height, width, _ = processor.original.shape
            fba_image = processor.original.copy()
            fba_trimap = trimap
            resize_factor = 4 if height > 4000 or width > 4000 else 2
            fba_image = cv2.resize(fba_image, (width // resize_factor, height // resize_factor))
            fba_trimap = cv2.resize(fba_trimap, (width // resize_factor, height // resize_factor))
            _, _, alpha = get_composite_image(fba_image, fba_trimap)
            alpha = cv2.resize(alpha, (width, height))
            foreground_mask = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            # The preview uses the SkySegNet mask only, without FBA Matting.
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            foreground_mask = cv2.bitwise_not(mask)
        sky_rgb = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB)
        output = processor.process_final_image(
            processor.original,
            foreground_mask,
            sky_rgb,
            data["tint_val"],
            data["mod_val"],
        )

        success, encoded = cv2.imencode(".jpg", output)
        if not success:
            return Response(
                {"detail": "Could not encode preview image."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return FileResponse(
            BytesIO(encoded.tobytes()),
            content_type="image/jpeg",
            as_attachment=False,
            filename="sky-preview.jpg",
        )
