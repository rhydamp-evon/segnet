from atlentis_ml_api.settings import fba_model
from core.networks.fba_matting.transforms import trimap_transform, groupnorm_normalise_image
from core.networks.fba_matting import build_model

import numpy as np
import cv2
import torch
from datetime import datetime


def read_image(image):
    image = (image / 255.0)
    return image[:, :, ::-1]


def read_trimap(image):
    trimap_im = image / 255.0
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap


def np_to_torch(x):
    if torch.cuda.is_available():
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cpu()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    ''' Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    '''
    h, w = trimap_np.shape[:2]

    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        np_to_torch_time = datetime.now()
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)
        # print('Time after np to torch ', datetime.now() - np_to_torch_time)

        trimap_transformed_torch = np_to_torch(
            trimap_transform(trimap_scale_np))

        image_transformed_torch = groupnorm_normalise_image(
            image_torch.clone(), format='nchw')

        fba_model_prediction_start = datetime.now()
        output = model(image_torch, trimap_torch,
                       image_transformed_torch, trimap_transformed_torch)
        # print("Time for fba model prediction ",
        #       datetime.now() - fba_model_prediction_start)
        output = cv2.resize(output[0].cpu().numpy().transpose(
            (1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)

    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]
    return fg, bg, alpha


def get_composite_image(image, trimap):
    image = read_image(image)
    trimap = read_trimap(trimap)

    # transform two channel trimap back to single channel
    trimap_im = trimap[:, :, 1] + (1-np.sum(trimap, -1))/2
    pred_started = datetime.now()
    # print("Preding FBA ..")
    fg, bg, alpha = pred(image, trimap, fba_model)
    # print("Time for FBA prediction ", datetime.now() - pred_started)
    return fg*255, bg*255, alpha*255
