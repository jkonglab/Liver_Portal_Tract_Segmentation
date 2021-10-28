# -*- coding: utf-8 -*-

import os
import random
import torch
import cv2
import torch.nn as nn
import torchvision as tv
from glob import glob
from torch.utils.data import Dataset


class ConvBlock(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1,
                 padding_mode="zeros", bias=False, activation='leaky'):
        super().__init__()

        if activation == 'leaky':
            act_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif activation == 'relu':
            act_layer = nn.Relu(inplace=True)
        else:
            act_layer = nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      padding_mode=padding_mode, bias=bias, groups=groups),
            nn.BatchNorm2d(out_ch),
            act_layer
        )

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    """Deconvolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 padding_mode="zeros", bias=False):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                               padding_mode=padding_mode, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DepthwiseSeparableConvBlock(nn.Module):
    """Depth-wise Seperable Convolution Layer"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            ConvBlock(in_ch, in_ch, kernel_size, groups=in_ch, padding=padding, padding_mode="reflect", activation='none'),
            ConvBlock(in_ch, out_ch, 1)
        )

    def forward(self, x):
        return self.net(x)


class DoubleConvBlock(nn.Module):
    """Two Convolution Layers"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch, kernel_size, stride=1, padding=padding, padding_mode="reflect"),
            ConvBlock(out_ch, out_ch, kernel_size, stride=1, padding=padding, padding_mode="reflect")
        )

    def forward(self, x):
        return self.net(x)


class DoubleDWSCBlock(nn.Module):
    """Two Depth-wise Seperable Convolution Layers"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        self.net = nn.Sequential(
            DepthwiseSeparableConvBlock(in_ch, out_ch, kernel_size),
            DepthwiseSeparableConvBlock(out_ch, out_ch, kernel_size)
        )

    def forward(self, x):
        return self.net(x)


class UpConvBlock(nn.Module):
    """Up-Sampling Layer plus Convolution Layer"""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect")
        )

    def forward(self, x):
        return self.net(x)


class ChannelPool(nn.Module):
    """Calculating Channel-wise Max-pooling and Mean-pooling"""

    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()

        self.compress = ChannelPool()
        self.spatial = ConvBlock(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, activation='relu')

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class ResBlock(nn.Module):
    """Residual Block with One Depth-wise Seperable Convolution Layers and One Spatial Gate"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        self.net = nn.Sequential(
            DepthwiseSeparableConvBlock(in_ch, out_ch, kernel_size),
            SpatialGate()
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class Res2Block(nn.Module):
    """Residual Block with Two Depth-wise Seperable Convolution Layers and One Spatial Gate"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        self.net = nn.Sequential(
            DoubleDWSCBlock(in_ch, out_ch, kernel_size),
            SpatialGate(kernel_size=kernel_size)
        )
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)
