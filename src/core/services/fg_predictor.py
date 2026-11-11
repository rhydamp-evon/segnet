import os
import torch
import cv2
import numpy as np

from core.networks.segmentation import SkySegNet

import torchvision.transforms.functional as TF
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DataTransform:
    def __init__(self, height=256, width=256):
        self.width = width
        self.height = height

    def preprocess(self, img):
        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.height, self.width], interpolation=transforms.functional.InterpolationMode.BICUBIC)
        # mask = TF.to_pil_image(mask)
        # mask = TF.resize(mask, [self.height, self.width], interpolation=transforms.functional.InterpolationMode.BICUBIC)
        img = TF.to_tensor(img)
        # mask = TF.to_tensor(mask)
        return img

class FGPredictor:
  def __init__(self, device='cpu', checkpoint='checkpoint2.pth'):
    self.device = device 
    self.model = self.load_model(checkpoint)

  def load_model(self, chechpoint_path):
    checkpoint = torch.load(chechpoint_path, map_location=torch.device(self.device))
    # model = UNET(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(self.device)
    #   if chechpoint_path == 'best_ckpt.pth':
    #     model = define_G(input_nc=3, output_nc=1, ngf=64, netG='coord_resnet50').to(self.device)
    #     model.load_state_dict(checkpoint["model_G_state_dict"])
    #   else:
    model = SkySegNet(in_channels=3, out_channels=1).to(self.device)
    model.load_state_dict(checkpoint["state_dict"])
    return model

  def _transform(self, img, height=256, width=256):
    transform = DataTransform(height=height, width=width)
    img = transform.preprocess(img)
    return img

  def check_location(self, loc, outdir):
    path = os.path.join(loc, outdir)
    if not os.path.exists(path):
      os.mkdir(path)

  def retrieve_foreground(self, image, res):
    self.model.eval()
    height, width, _ = image.shape
    image = self._transform(image, res, res)
    image = image.to(device=self.device)
    image = image.unsqueeze(0)
    with torch.no_grad():
      pred = self.model(image)
      pred = TF.resize(pred, [height, width], interpolation=2)
      return pred