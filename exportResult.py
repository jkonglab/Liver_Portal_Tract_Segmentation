import os
import cv2
import math
import openslide
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from glob import glob
from datetime import datetime
from model import *


@torch.no_grad()
def predictImage(file, patch_size, patch_margin, ratio, model, device):

    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # read image coefficiences
    [width, height] = file.dimensions

    # distinguish foreground and background (using high level image)
    level = 2
    factor = file.level_downsamples[level]
    im_hl = np.array(file.read_region((0, 0), level, file.level_dimensions[level]))
    hsv = cv2.cvtColor(im_hl, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    fg = (s > 100) * 1.0

    # # predict result by patch
    size = int(patch_size * ratio)
    margin = int(patch_margin * ratio)
    rows = math.ceil(height / size)
    cols = math.ceil(width / size)
    mask = np.zeros((rows*size, cols*size), np.uint8)
    for i in range(0, rows):
        for j in range(0, cols):
            sf = size / factor
            fg_patch = fg[int(i*sf):int((i+1)*sf), int(j*sf):int((j+1)*sf)]
            if np.sum(fg_patch, axis=(0, 1)) > 100:
                im_patch = np.array(file.read_region((j*size-margin, i*size-margin), 0, (size+2*margin, size+2*margin)))
                im_patch = cv2.cvtColor(im_patch, cv2.COLOR_RGB2BGR)
                im_patch = cv2.resize(im_patch, (patch_size+2*patch_margin, patch_size+2*patch_margin))
                X = transforms(im_patch).to(device)
                X.unsqueeze_(0)
                Y = unet(X)
                patch_prob = Y.squeeze().cpu().numpy()
                patch_prob = cv2.resize(patch_prob, (size+2*margin, size+2*margin))
                smoothed = cv2.blur(patch_prob, (15, 15))
                patch_mask = smoothed[margin:size+margin, margin:size+margin] > 0
                mask[i*size:(i+1)*size, j*size:(j+1)*size] = patch_mask * 255

    mask = mask[0:height, 0:width]
    output_mask = cv2.resize(mask, None, fx=0.125, fy=0.125)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    selected_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 2e4:
            selected_contours.append(contour)
    return output_mask, selected_contours


def exportContours(file, contours, mpp):
    with open(file, 'w') as f:
        f.write("<Annotations MicronsPerPixel=\"{:.6f}\">\n".format(mpp))
        f.write("\t<Annotation Id=\"1\" Name=\"Results\" ReadOnly=\"0\" NameReadOnly=\"0\" LineColorReadOnly=\"0\" "
                "Incremental=\"0\" Type=\"4\" LineColor=\"65280\" Visible=\"1\" Selected=\"0\" MarkupImagePath=\"\" "
                "MacroName=\"\">\n")
        f.write("\t\t<Attributes>\n\t\t\t<Attribute Name=\"Description\" Id=\"0\" Value=\"\"/>\n\t\t</Attributes>\n")
        f.write("\t\t<Regions>\n\t\t\t<RegionAttributeHeaders>\n\t\t\t\t<AttributeHeader Id=\"9999\" Name=\"Region\" "
                "ColumnWidth=\"-1\"/>\n\t\t\t\t<AttributeHeader Id=\"9997\" Name=\"Length\" "
                "ColumnWidth=\"-1\"/>\n\t\t\t\t<AttributeHeader Id=\"9996\" Name=\"Area\" "
                "ColumnWidth=\"-1\"/>\n\t\t\t\t<AttributeHeader Id=\"9998\" Name=\"Text\" "
                "ColumnWidth=\"-1\"/>\n\t\t\t\t<AttributeHeader Id=\"1\" Name=\"Description\" "
                "ColumnWidth=\"-1\"/>\n\t\t\t</RegionAttributeHeaders>\n")
        num = 0
        for contour in contours:
            num = num + 1
            length = cv2.arcLength(contour.astype(np.float32), True)
            area = cv2.contourArea(contour.astype(np.float32))
            f.write("\t\t\t<Region Id=\"{:d}\" Type=\"1\" Zoom=\"0.360000\" Selected=\"0\" ImageLocation=\"\" ImageFocus"
                    "=\"-1\" Length=\"{:.1f}\" Area=\"{:.1f}\" LengthMicrons=\"{:.1f}\" AreaMicrons=\"{:.1f}\" "
                    "Text=\"\" NegativeROA=\"0\" InputRegionId=\"0\" Analyze=\"1\" DisplayId=\"{:d}\">\n\t\t\t\t"
                    "<Attributes/>\n\t\t\t\t<Vertices>\n".format(num, length, area, length*mpp, area*mpp*mpp, num))
            rows = contour.shape[0]
            for i in range(rows):
                f.write("\t\t\t\t\t<Vertex X=\"{:d}\" Y=\"{:d}\" Z=\"0\"/>\n".format(contour[i, 0, 0], contour[i, 0, 1]))
            f.write("\t\t\t\t</Vertices>\n\t\t\t</Region>\n")
        f.write("\t\t</Regions>\n\t\t<Plots/>\n\t</Annotation>\n</Annotations>")



if __name__ == '__main__':
    stop_names = []
    mpp_standard = 0.2478

    # set network
    device = torch.device('cuda:0')
    unet = UNet_Res_Multi(32, 3, 1).eval()
    unet.to(device)
    unet.load_state_dict(torch.load(opt.load_model, map_location=device))

    for path in glob('../../data/origin/*.svs'):
        file = os.path.basename(path)
        name = os.path.splitext(file)[0]

        if name in stop_names:
            continue

        img = openslide.open_slide(path)
        prop = img.properties
        [width, height] = img.dimensions
        mpp = float(prop['aperio.MPP'])
        mask, contours = predictImage(img, 1000, 140, mpp_standard/mpp, unet, device)

        result_file = 'results_res_multi/' + name + '.xml'
        # print(img.dimensions, result_file)
        exportContours(result_file, contours, mpp)
        print("Image {} finished at {}".format(name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
