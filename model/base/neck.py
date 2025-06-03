import torch
import torch.nn as nn
from .components import Conv, C2f


def runUpsample(x):
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        self.l12 = C2f(int(512 * w * (1+r)), int(512 * w), n, False)
        self.l15 = C2f(int(768 * w), int(256 * w), n, False)
        self.l18 = C2f(int(768 * w), int(512 * w), n, False)
        self.l21 = C2f(int(512 * w * (1+r)), int(512 * w * r), n, False)

        self.l16 = Conv(int(256 * w), int(256 * w), 3, 2, 1)
        self.l19 = Conv(int(512 * w), int(512 * w), 3, 2, 1)

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape for head:
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        # Top-down path
        topDownlayer_2 = torch.cat([runUpsample(feat3), feat2], dim = 1)
        topDownlayer_2 = self.l12(topDownlayer_2)

        topDownlayer_1 = torch.cat([runUpsample(topDownlayer_2), feat1], dim = 1)
        X = self.l15(topDownlayer_1)

        # Bottom-up path
        bottomUplayer_0 = self.l16(X)
        bottomUplayer_0 = torch.cat([bottomUplayer_0, topDownlayer_2], dim = 1)
        Y = self.l18(bottomUplayer_0)

        bottomUplayer_1 = self.l19(Y)
        bottomUplayer_1 = torch.cat([bottomUplayer_1, feat3], dim = 1)
        Z = self.l21(bottomUplayer_1)

        # The first return value 'topDownlayer_2' is an intermediate feature map.
        # The subsequent unpacking in yolomodel.py as `_, X, Y, Z = self.neck.forward(...)`
        # correctly ignores it and passes X, Y, Z to the head.
        return topDownlayer_2, X, Y, Z
