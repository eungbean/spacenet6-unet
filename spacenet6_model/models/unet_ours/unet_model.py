""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Flatten, Activation
from segmentation_models_pytorch.base import SegmentationModel

class UNet(SegmentationModel):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.encoder = UnetEncoder(n_channels)
        self.decoder = UnetDecoder(n_classes)
        self.segmentation_head = SegmentationHead(in_channels=64, out_channels=n_classes, activation='sigmoid', kernel_size=3)
        self.classification_head = None
        self.initialize()
        

class UnetEncoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        bilinear = True
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        features = [x1,x2,x3,x4,x5]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class UnetDecoder(nn.Module):
    def __init__(self, n_classes, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = True

        factor = 2 if self.bilinear else 1
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, *features):
        x1, x2, x3, x4, x5 = features
        
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        return out

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)