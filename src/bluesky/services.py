import logging
import sys
import traceback
import cv2
import requests
import json
from django.core.files.base import ContentFile
from sentry_sdk import capture_exception
from atlentis_ml_api.settings import (
    BLUESKY_WEBHOOK_URL,
    BLUESKY_WEBHOOK_API_KEY,
    APP_ENV,
)
from bluesky.models import (
    BlueSkyRequest,
    BlueSkyVariant,
    SkyConfig,
    BlueSkyProcessed,
    BlueSkyProcessFailedLog,
)
from core.utils import BlueSkyProcessor, read_image_from_url

logger = logging.getLogger("error_logger")


def get_param_values(data, sky_config):
    if "tint_val" in data:
        tint_val = data.pop("tint_val")
    else:
        tint_val = sky_config.tint_val
    if "mod_val" in data:
        mod_val = data.pop("mod_val")
    else:
        mod_val = sky_config.mod_val
    return tint_val, mod_val, data


def create_bluesky_request(data):
    try:
        sky_variant = SkyConfig.objects.order_by("order").first()
        tint_val, mod_val, data = get_param_values(data, sky_variant)
        bluesky_req = BlueSkyRequest.objects.create(**data)
        img = read_image_from_url(bluesky_req.original_image_path)
        bluesky_req.is_image_downloaded = True
        bluesky_req.save(update_fields=["is_image_downloaded"])
        processor = BlueSkyProcessor(image=img)
        # Pull very first Sky Config in order
        output, bluesky_req = processor.create_thumnail_image(
            bluesky_req, sky_variant, tint_val, mod_val
        )
        bluesky_variant = BlueSkyVariant(
            blue_sky_request=bluesky_req,
            sky_config=sky_variant,
            is_processed=True,
        )
        _, buf = cv2.imencode(".jpg", output)  # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        bluesky_variant.relative_path_processed.save("RES.jpg", content)
        return bluesky_req, bluesky_variant
    except Exception as e:
        logger.exception(e)
        capture_exception(e)


############## Service to create new blue sky variant ##############
def create_bluesky_variant(data):
    try:
        bluesky_variant, created = BlueSkyVariant.objects.get_or_create(
            blue_sky_request_id=data["blue_sky_request_id"],
            sky_config_id=data["sky_config_id"],
        )

        if not created and bluesky_variant.relative_path_processed:
            tint_val, mod_val, data = get_param_values(data, bluesky_variant.sky_config)
            if (
                bluesky_variant.tint_val == tint_val
                and bluesky_variant.mod_val == mod_val
            ):
                return bluesky_variant
            else:
                bluesky_req = bluesky_variant.blue_sky_request
                sky_config = bluesky_variant.sky_config
        else:
            bluesky_req = BlueSkyRequest.objects.get(id=data["blue_sky_request_id"])
            sky_config = SkyConfig.objects.get(id=data["sky_config_id"])
            tint_val, mod_val, data = get_param_values(data, sky_config)
        img = read_image_from_url(bluesky_req.original_image_path)
        processor = BlueSkyProcessor(image=img)
        output, bluesky_req = processor.create_thumnail_image(
            bluesky_req, sky_config, tint_val, mod_val
        )
        _, buf = cv2.imencode(".jpg", output)  # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        bluesky_variant.is_processed = True
        bluesky_variant.relative_path_processed.save("RES.jpg", content)
        return bluesky_variant
    except Exception as e:
        capture_exception(e)


############## Service to process final Image ##############
class BlueSkyProcessedService:
    def __init__(self, data):
        self.data = data

    def get_bluesky_process(self, bluesky_req):
        prediction_mask_path = None
        composite_image_path = None
        relative_path_processed = None
        is_processed = False
        bluesky_process = BlueSkyProcessed.objects.filter(
            platform_id=bluesky_req.platform_id,
            content_id=bluesky_req.content_id,
            sky_config_id=self.data["sky_config_id"],
        )[:1]
        if len(bluesky_process) > 0:
            bluesky_process = bluesky_process[0]
            if bluesky_process.blue_sky_request == bluesky_req:
                return bluesky_process
            prediction_mask_path = bluesky_process.prediction_mask_path
            composite_image_path = bluesky_process.composite_image_path
            relative_path_processed = bluesky_process.relative_path_processed
            is_processed = True
        else:
            bluesky_process = BlueSkyProcessed.objects.filter(
                platform_id=bluesky_req.platform_id,
                content_id=bluesky_req.content_id,
            )[:1]
            if len(bluesky_process) > 0:
                bluesky_process = bluesky_process[0]
                prediction_mask_path = bluesky_process.prediction_mask_path
                composite_image_path = bluesky_process.composite_image_path
        bluesky_process = BlueSkyProcessed.objects.create(
            platform_id=bluesky_req.platform_id,
            content_id=self.data["content_id"],
            project_id=bluesky_req.project_id,
            content_type=bluesky_req.content_type,
            blue_sky_request=bluesky_req,
            original_image_path=bluesky_req.original_image_path,
            sky_config_id=self.data["sky_config_id"],
            prediction_mask_path=prediction_mask_path,
            composite_image_path=composite_image_path,
            relative_path_processed=relative_path_processed,
            is_processed=is_processed,
        )
        return bluesky_process

    def push_results_to_api(self, bluesky_process):
        if APP_ENV == "dev":
            return
        data = {
            "project_id": bluesky_process.project_id,
            "platform_id": bluesky_process.platform_id,
            "content_id": bluesky_process.content_id,
            "content_type": bluesky_process.content_type,
            "s3_relative_path": bluesky_process.relative_path_processed.url,
        }
        headers = {
            "Api-Key": BLUESKY_WEBHOOK_API_KEY,
            "Content-Type": "application/json",
        }
        res = requests.post(BLUESKY_WEBHOOK_URL, data=json.dumps(data), headers=headers)
        if res.status_code == 200:
            bluesky_process.is_pushed_to_api = True
            bluesky_process.save(update_fields=("is_pushed_to_api", "updated_at"))

    def process_image(self):
        bluesky_req = BlueSkyRequest.objects.get(id=self.data["blue_sky_request_id"])
        bluesky_process = self.get_bluesky_process(bluesky_req)

        try:
            if not bluesky_process.is_processed:
                # gaining tint and mod values
                tint_val, mod_val, _ = get_param_values(
                    self.data, bluesky_process.sky_config
                )
                img = read_image_from_url(bluesky_process.original_image_path)
                processor = BlueSkyProcessor(image=img)
                output, bluesky_process = processor.create_processed_image(
                    bluesky_process,
                    bluesky_process.sky_config,
                    bluesky_process.composite_image_path,
                    tint_val,
                    mod_val,
                )
                # cropped_image: cv2 / np array
                _, buf = cv2.imencode(".jpg", output)
                bluesky_process.is_processed = True
                content = ContentFile(buf.tobytes())
                bluesky_process.relative_path_processed.save("FINAL.png", content)
            if not bluesky_process.is_pushed_to_api:
                self.push_results_to_api(bluesky_process)
        except Exception as e:
            _, _, stacktrace = sys.exc_info()
            BlueSkyProcessFailedLog.objects.create(
                blue_sky_processed=bluesky_process,
                error=str(e),
                stack_trace="".join(traceback.format_tb(stacktrace)),
            )
            return False
        return True
