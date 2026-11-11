import logging
import cv2
import numpy as np
import torchvision
import torch
import requests
from PIL import Image
from django.core.files.base import ContentFile

from atlentis_ml_api.settings import USE_S3, predictor
from core.services.color_refinement import color_transfer
from core.services.matting import get_composite_image, fba_model
from core.filters import guided_upsample

logger = logging.getLogger("default_logger")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# predictor = FGPredictor(device=DEVICE, checkpoint=SEGMENTATION_MODEL)
# predictor = None


def resize_image(image, ref, beta=0.01):
    """
    Returns resized image to reference image.
    """
    h, w, _ = ref.shape
    h1, w1, _ = image.shape
    h_rat, w_rat = h / h1, w / w1
    if h_rat < w_rat:
        r = w_rat + beta
    else:
        r = h_rat + beta
    bg_h, bg_w = int(h1 * r), int(w1 * r)
    image = cv2.resize(image, (bg_w, bg_h))
    return image


def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 7)
    img_blur = img_blur.astype(np.uint8)
    img_canny = cv2.Canny(img_blur, 0, 42)
    kernel = np.ones((19, 19))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    img_erode = cv2.erode(img_dilate, kernel, iterations=4)
    return img_erode


def find_grad_height(img):
    contours, hierarchies = cv2.findContours(
        process(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.004 * peri, True)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        # cv2_imshow(img)
        COLOR = (0, 0, 255)
        indices = np.where(np.all(img == COLOR, axis=-1))
        indexes = list(zip(indices[0], indices[1]))
        cord_min = max(indexes)  # lowest Pixel on Longest contour
        cord_max = min(indexes)
        h, w, _ = img.shape

        height_min = cord_min[0] if h - cord_min[0] < 20 else cord_min[0] + 20

        height_max = cord_max[0] if cord_max[0] < 20 else cord_max[0] - 20

        img = cv2.line(img, (0, height_min), (w, height_min), (0, 255, 0), thickness=5)
        img = cv2.line(img, (0, height_max), (w, height_max), (0, 255, 0), thickness=5)
        if height_min < 0:
            height_min = 0
        if height_max < 0:
            height_max = 0
        return height_min, height_max, img, True
    return None, None, None, False


def vertical_grad(src, color_start, color_end):
    h = src.shape[0]

    # Create a transparent picture with the same size as the original picture
    grad_img = np.ndarray(src.shape, dtype=np.uint8)

    # opencv defaults to BGR format instead of RGB format
    g_b = float(color_end[0] - color_start[0]) / h
    g_g = float(color_end[1] - color_start[1]) / h
    g_r = float(color_end[2] - color_start[2]) / h

    for i in range(h):
        for j in range(src.shape[1]):

            grad_img[i, j, 0] = color_start[0] + i * g_b
            grad_img[i, j, 1] = color_start[1] + i * g_g
            grad_img[i, j, 2] = color_start[2] + i * g_r
    return grad_img


def image_blending(image, mask, background, tint_val=100, mod_val=50, identifier=None):
    """
    Returns final image after applying gradient, image blending and color refactoring.
    """
    logger.debug(
        f"{identifier}: Initial shapes in image blending - Mask: {mask.shape}, Image: {image.shape}"
    )
    h, w, _ = image.shape
    background = resize_image(background, image)
    bh, bw, _ = background.shape
    margin = (bw - w) // 2

    background = background[0:h, margin : bw - margin]
    mask_for_cnt = mask.copy()
    img_bkp = image.copy()
    # Get starting point for horizon.
    logger.debug(f"{identifier}: Applying gradient...")
    height_low, height_max, mask_cnt, contour_found = find_grad_height(mask_for_cnt)
    if contour_found:
        top_val = h - ((height_low - height_max) + (h - height_low))

        """
        Create a buffer so that the horizon could be visible properly.
        """
        lh = height_low + (h - height_low) // 2
        lh = h - lh
        bf_buffer = cv2.copyMakeBorder(
            background, 0, lh, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        background = bf_buffer[lh:, 0:w]

        background_mask = np.ndarray((height_low - height_max, w, 3), dtype=np.uint8)

        pixel_val = int((height_low - height_max) * 255 / height_low)
        grad = vertical_grad(
            background_mask, (0, 0, 0), (pixel_val, pixel_val, pixel_val)
        )
        bottom_val = h - height_low
        if bottom_val < 0:
            bottom_val = 0
        grad = cv2.copyMakeBorder(
            grad, top_val, bottom_val, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        background = cv2.addWeighted(background, 1.0, grad, 0.6, 0.0)
        image = img_bkp
        # Apply Color transfer
        logger.debug(f"{identifier}: Applying color transfer...")
        image = color_transfer(
            image[:, :, ::-1], background[:, :, ::-1], tint_val, mod_val
        )
        logger.debug(f"{identifier}: Color transfer done...")
        image = image.astype(float)
        background = background.astype(float)
        mask = mask.astype(float) / 255
        logger.debug(
            f"{identifier}: Shapes => Mask {mask.shape}, Image {image.shape}, Background {background.shape}"
        )
        image = cv2.multiply(mask, image)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - mask, background)
        # Add the masked foreground and background.
        out_image = cv2.add(image, background)
        logger.debug(f"{identifier}: Image blending done...")
        return out_image, mask_cnt
    logger.debug(f"{identifier}: Contours not found.")
    return image, mask


def read_image_from_url(url):
    """Read image from url and save it in memory"""
    req = requests.get(url)
    img_array = np.asarray(bytearray(req.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def read_image_from_object(image, param=cv2.IMREAD_COLOR):
    if USE_S3:
        return read_image_from_url(image.url)
    img = cv2.imread(image.path, param)
    return img


class BlueSkyProcessor:
    def __init__(self, image):
        self.image = image
        self.original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.identifier = None

    def read_background_image(self, sky_variant, height=0, width=0, size=0):
        # Size > 0. thumbnail,  1. 2k, 2. original
        resize = False
        background = sky_variant
        if (
            height <= background.sky_image_thumbnail_height
            and width <= background.sky_image_thumbnail_width
        ):
            background_url = background.sky_image_thumbnail_path
        elif (
            height <= background.sky_image_2k_height
            and width <= background.sky_image_2k_width
        ):
            background_url = background.sky_image_2k_path
        elif (
            height <= background.sky_image_fullsize_height
            and width <= background.sky_image_fullsize_width
        ):
            background_url = background.sky_image_fullsize_path
        else:
            if height > width and background.sky_image_fullsize_vertical_path != "":
                background_url = background.sky_image_fullsize_vertical_path
            else:
                background_url = background.sky_image_fullsize_path
                resize = True
                logger.debug(
                    f"{self.identifier}: Image is too big from existing background images"
                )
        background = read_image_from_object(background_url)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        if resize:
            self.original = resize_image(self.original, background, beta=0.0)
        return background

    def resize_to_thumbnail_image(self, sky_config, height, width):
        max_resize_ratio = 4
        # print(sky_config.sky_image_thumbnail_height, "-", sky_config.sky_image_thumbnail_width)
        for i in range(1, max_resize_ratio):
            if (
                height // i <= sky_config.sky_image_thumbnail_height
                and height // i <= sky_config.sky_image_thumbnail_width
            ):
                self.original = cv2.resize(self.original, (width // i, height // i))
                logger.debug(
                    f"{self.identifier}: Shape after resizing: {self.original.shape}"
                )
                return
        else:
            self.original = cv2.resize(
                self.original, (width // max_resize_ratio, height // max_resize_ratio)
            )
        logger.debug(f"{self.identifier}: Shape after resizing: {self.original.shape}")

    def create_mask(self, bluesky_request=None):
        if bluesky_request and bluesky_request.prediction_mask_path:
            mask = read_image_from_object(
                bluesky_request.prediction_mask_path, param=cv2.IMREAD_GRAYSCALE
            )
            return None, mask
        mask = predictor.retrieve_foreground(self.original, 768)
        grid = torchvision.utils.make_grid(mask)
        # # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        mask = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        mask2 = mask[..., 0] / 255.0
        image = self.original / 255.0
        kernel_size = 8
        filtered, mask = guided_upsample(image, mask2, kernel_size)
        return filtered * 255, (mask * 255).round().astype(np.uint8)

    def process_final_image(self, image, mask, background, tint_val=100, mod_val=50):
        out_image, _ = image_blending(
            image, mask, background, tint_val, mod_val, self.identifier
        )
        return out_image[:, :, ::-1]

    def create_thumnail_image(
        self, bluesky_request, sky_variant, tint_val=100, mod_val=50
    ):
        self.identifier = f"Thumbnail Req-{bluesky_request.id}-SV-{sky_variant.id}"
        h, w, _ = self.original.shape
        self.resize_to_thumbnail_image(sky_variant, h, w)
        # background = self.read_background_image(sky_variant, height=h, width=w)
        if not bluesky_request.prediction_mask_path:
            logger.debug(f"{self.identifier}: Generating mask...")
            _, mask = self.create_mask(bluesky_request=bluesky_request)
            # cropped_image: cv2 / np array
            _, buf = cv2.imencode(".jpg", mask)
            content = ContentFile(buf.tobytes())
            bluesky_request.prediction_mask_path.save("MASK.png", content)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            logger.debug(f"{self.identifier}: Using pre-generated mask...")
            mask = read_image_from_object(bluesky_request.prediction_mask_path)
        h, w, _ = self.original.shape
        background = self.read_background_image(sky_variant, height=h, width=w)
        mask = cv2.bitwise_not(mask)
        return (
            self.process_final_image(
                self.original, mask, background, tint_val, mod_val
            ),
            bluesky_request,
        )

    def create_processed_image(
        self, bluesky_process, sky_variant, composite=None, tint_val=100, mod_val=50
    ):
        self.identifier = (
            f"Process Req-{bluesky_process.blue_sky_request_id}-SV-{sky_variant.id}"
        )
        h, w, _ = self.original.shape
        logger.debug(f"{self.identifier}: Height - {h}, Width - {w}")
        background = self.read_background_image(sky_variant, height=h, width=w)
        h, w, _ = self.original.shape
        if not composite:
            logger.debug(f"{self.identifier}: Creating composite image...")
            trimap, mask = self.create_mask()
            logger.debug(f"{self.identifier}: Trimap generated.")
            image = self.original.copy()
            if h > 4000 or w > 4000:
                image = cv2.resize(image, (w // 4, h // 4))
                trimap = cv2.resize(trimap, (w // 4, h // 4))
                logger.debug(f"{self.identifier}: Resized by factor 4.")
            else:
                image = cv2.resize(image, (w // 2, h // 2))
                trimap = cv2.resize(trimap, (w // 2, h // 2))
                logger.debug(f"{self.identifier}: Resized by factor 2.")

            _, _, composite = get_composite_image(image, trimap)
            logger.debug(f"{self.identifier}: Composite image generated.")
            _, buf = cv2.imencode(".jpg", mask)
            content = ContentFile(buf.tobytes())
            bluesky_process.prediction_mask_path.save("mask.jpg", content)
            _, buf = cv2.imencode(".jpg", composite)
            content = ContentFile(buf.tobytes())
            bluesky_process.composite_image_path.save("composite.jpg", content)
            composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2RGB)
        else:
            logger.debug(f"{self.identifier}: Reading pre-generated composite image.")
            composite = read_image_from_object(bluesky_process.composite_image_path)
        composite = cv2.resize(composite, (w, h))
        return (
            self.process_final_image(
                self.original, composite, background, tint_val, mod_val
            ),
            bluesky_process,
        )

    # def create_demo_image(self, bluesky_demo):
    #     if bluesky_demo and bluesky_demo.prediction_mask_path:
    #         mask = read_image_from_object(
    #             bluesky_demo.prediction_mask_path)
    #     else:
    #         _, mask = self.create_mask(bluesky_request=bluesky_request)
    #         # cropped_image: cv2 / np array
    #         _, buf = cv2.imencode('.jpg', mask)
    #         content = ContentFile(buf.tobytes())
    #         bluesky_demo.prediction_mask_path.save('mask.jpg', content)
    #         mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# def reduce_model_size():
#     from atlentis_ml_api.settings import SEGMENTATION_MODEL
#     import torch
#     _checkpoint = torch.load(SEGMENTATION_MODEL, map_location=torch.device('cpu'))
#     checkpoint = {
#         "state_dict": _checkpoint["state_dict"]
#     }
#     torch.save(checkpoint, 'test.pth')
