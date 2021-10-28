import torch.nn as nn
import torch
from utils import *


class UNet_Res_Multi(nn.Module):
    """MUSA-UNet Model"""

    def __init__(self, n_f, in_ch=3, out_ch=3):
        super().__init__()

        filters = [n_f, n_f*2, n_f*4, n_f*8, n_f*16]

        self.inconv = ConvBlock(in_ch, filters[0], 7, padding=3, padding_mode="reflect")

        self.eres1 = ResBlock(filters[0], filters[0], kernel_size=9)
        self.eres2 = ResBlock(filters[0], filters[1], kernel_size=7)
        self.eres3 = ResBlock(filters[1], filters[2], kernel_size=5)
        self.eres4 = ResBlock(filters[2], filters[3], kernel_size=3)
        self.eres5 = ResBlock(filters[3], filters[4], kernel_size=3)

        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
        self.mp3 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(2)

        self.uc1 = UpConvBlock(filters[4], filters[3])
        self.uc2 = UpConvBlock(filters[3], filters[2])
        self.uc3 = UpConvBlock(filters[2], filters[1])
        self.uc4 = UpConvBlock(filters[1], filters[0])

        self.dres1 = ResBlock(filters[4], filters[3], kernel_size=3)
        self.dres2 = ResBlock(filters[3], filters[2], kernel_size=3)
        self.dres3 = ResBlock(filters[2], filters[1], kernel_size=3)
        self.dres4 = ResBlock(filters[1], filters[0], kernel_size=3)

        self.outconv1 = ConvBlock(filters[4], 1, 1)
        self.outconv2 = ConvBlock(filters[3], 1, 1)
        self.outconv3 = ConvBlock(filters[2], 1, 1)
        self.outconv4 = ConvBlock(filters[1], 1, 1)
        self.outconv5 = ConvBlock(filters[0], 1, 1)

        self.up1 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.up2 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.up3 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.outconv = nn.Conv2d(5, out_ch, 1, stride=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ini = self.inconv(x)

        # Down sampling
        e1 = self.eres1(ini)
        m1 = self.mp1(e1)
        e2 = self.eres2(m1)
        m2 = self.mp2(e2)
        e3 = self.eres3(m2)
        m3 = self.mp3(e3)
        e4 = self.eres4(m3)
        m4 = self.mp4(e4)
        e5 = self.eres5(m4)

        # Up sampling
        u1 = self.uc1(e5)
        d1 = self.dres1(torch.cat([e4, u1], dim=1))
        u2 = self.uc2(d1)
        d2 = self.dres2(torch.cat([e3, u2], dim=1))
        u3 = self.uc3(d2)
        d3 = self.dres3(torch.cat([e2, u3], dim=1))
        u4 = self.uc4(d3)
        d4 = self.dres4(torch.cat([e1, u4], dim=1))

        # output
        o1 = self.outconv1(e5)
        o2 = self.outconv2(d1)
        o3 = self.outconv3(d2)
        o4 = self.outconv4(d3)
        o5 = self.outconv5(d4)

        out = torch.cat([self.up1(o1), self.up2(o2), self.up3(o3), self.up4(o4), o5], dim=1)
        return self.outconv(out)
