from sentry_sdk.api import capture_exception
from atlentis_ml_api.settings import USE_S3
from core.utils import BlueSkyProcessor, read_image_from_url, read_image_from_object, image_blending
import boto3
from core.services.fg_predictor import FGPredictor
import os
from atlentis_ml_api.settings import BASE_DIR, DEVICE
import torchvision
import torch
from core.filters import guided_upsample
import numpy as np
import cv2
from bluesky.models import SkyConfig
from core.services.matting import get_composite_image, fba_model
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from atlentis_ml_api.services import PushNotification
from core.utils import resize_image

if USE_S3:
    from atlentis_ml_api.settings import AWS_STORAGE_BUCKET_NAME, AWS_S3_CUSTOM_DOMAIN


class BlueSkyModelEvaluator:
    '''
        Evaluate model performace with the images present in the test_images_folder.
    '''

    def __init__(self, checkpoint) -> None:
        self.checkpoint_name = checkpoint
        self.checkpoint = os.path.join(
            BASE_DIR, 'checkpoints', checkpoint)
        self.predictor = FGPredictor(device=DEVICE, checkpoint=self.checkpoint)
        self.sky_variant = SkyConfig.objects.first()

    def get_test_images_urls_list(self):
        # Returns list of urls of the test images.
        s3 = boto3.resource('s3')

        my_bucket = s3.Bucket(AWS_STORAGE_BUCKET_NAME)
        images_list = []
        for object_summary in my_bucket.objects.filter(Prefix="assets/test-images/"):
            images_list.append(object_summary.key)
        return images_list

    def create_mask(self, original_image):
        mask = self.predictor.retrieve_foreground(original_image, 768)
        grid = torchvision.utils.make_grid(mask)
        # # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        mask = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        mask2 = mask[..., 0] / 255.0
        image = original_image / 255.0
        kernel_size = 8
        filtered, mask = guided_upsample(image, mask2, kernel_size)
        return filtered*255, (mask*255).round().astype(np.uint8)

    def read_background_image(self, sky_variant, original_image, height=0, width=0):
        # Size > 0. thumbnail,  1. 2k, 2. original
        resize = False
        background = sky_variant
        if (height <= background.sky_image_thumbnail_height and
                width <= background.sky_image_thumbnail_width):
            background_url = background.sky_image_thumbnail_path
        elif (height <= background.sky_image_2k_height and
              width <= background.sky_image_2k_width):
            background_url = background.sky_image_2k_path
        elif (height <= background.sky_image_fullsize_height and
              width <= background.sky_image_fullsize_width):
            background_url = background.sky_image_fullsize_path
        else:
            background_url = background.sky_image_fullsize_path
            resize = True
        background = read_image_from_object(background_url)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        if resize:
            original_image = resize_image(original_image, background, beta=0.0)
        return background, original_image

    def process_final_image(self, image, mask, background):
        out_image, _ = image_blending(image, mask, background)
        return out_image[:, :, ::-1]

    def resize_to_2k_image(self, original_image, sky_config):
        max_resize_ratio = 4
        height, width, _ = original_image.shape
        for i in range(1, max_resize_ratio):
            if (height // i <= sky_config.sky_image_2k_height 
                and height // i <=  sky_config.sky_image_2k_width):
                original_image = cv2.resize(original_image, (width//i, height//i))
                break
        else:
            original_image = cv2.resize(original_image, (width//max_resize_ratio, height//max_resize_ratio))
        return original_image

    def model_prediction(self, image_url):
        image = read_image_from_url(image_url)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # h, w, _ = original_image.shape
        print("Before adjusting to 2k:", original_image.shape)
        original_image = self.resize_to_2k_image(original_image, self.sky_variant)
        print("After adjusting to 2k:", original_image.shape)
        # original_image = cv2.resize(original_image, (w//2, h//2))
        trimap, mask = self.create_mask(original_image)
        return trimap, mask, original_image

    def final_processing(self, original_image, trimap):
        _, _, composite = get_composite_image(original_image, trimap)
        h, w, _ = original_image.shape
        # _, buf = cv2.imencode('.jpg', mask)
        composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2RGB)
        background, original_image = self.read_background_image(
            self.sky_variant, original_image, h, w)
        return self.process_final_image(original_image, composite, background)

    def save_np_image_as_file(self, image, file_key):
        _, buf = cv2.imencode('.jpg', image)
        content = ContentFile(buf.tobytes())
        default_storage.save(file_key, content)

    def evaluate(self):
        # driver method
        push_notification = PushNotification()
        # push_notification.model_evaluation_started()
        test_images = self.get_test_images_urls_list()
        for image_url in test_images:
            try:
                image_name = image_url.split("/")[-1]
                print("Evaluating - ", image_name)
                checkpoint_name = self.checkpoint_name.replace(".pth", "")
                file_key = f"model_eval/{checkpoint_name}/{image_name}"

                if image_url[-1] == "/":
                    continue
                aws_image_url = f"https://{AWS_S3_CUSTOM_DOMAIN}/{image_url}"
                trimap, mask, original_image = self.model_prediction(
                    aws_image_url)
                image = self.final_processing(
                    original_image, trimap)

                # file_name_to_save
                image_name = image_url.split("/")[-1]
                checkpoint_name = self.checkpoint_name.replace(".pth", "")

                image_file_key = f"model_eval/{checkpoint_name}/{image_name}"
                mask_file_key = f"model_eval/{checkpoint_name}/mask-{image_name}"

                self.save_np_image_as_file(image, image_file_key)
                self.save_np_image_as_file(mask, mask_file_key)
            except Exception as e:
                capture_exception(e)
        # Evaluation completed, send msg on slack.
        push_notification.model_evaluation_completed()
