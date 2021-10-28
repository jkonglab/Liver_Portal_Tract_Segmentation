import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from config import *
from model import *
from glob import glob


@torch.no_grad()
def predict(**kwargs):

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device('cpu')

    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    unet = UNet_Res_Multi(32, 3, 1).eval()
    unet.to(device)
    unet.load_state_dict(torch.load(opt.model_path, map_location=device))

    for path in glob(os.path.join(opt.test_path, '*.tif')):
        name = os.path.basename(path)
        img = cv2.imread(path)
        height, width, _ = img.shape
        height_pad = (32 - height % 32) % 32
        width_pad = (32 - width % 32) % 32
        img = cv2.copyMakeBorder(img, 0, height_pad, 0, width_pad, cv2.BORDER_CONSTANT, value=0)
        X = transforms(img).to(device)
        X.unsqueeze_(0)
        Y = torch.sigmoid(unet(X))
        prob = Y.squeeze().cpu().numpy()
        prob = prob[0:height, 0:width]
        output = (prob*255).astype(np.uint8)
        cv2.imwrite(os.path.join(opt.dst_path, name), output)


if __name__ == '__main__':
    predict(model_path='checkpoints/model_40.pth', dst_path='results')
