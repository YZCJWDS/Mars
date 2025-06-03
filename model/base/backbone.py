import torch.nn as nn
from .components import Conv, C2f, SPPF


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):

        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        self.l0 = Conv(3, int(64 * w), 3, 2, 1)
        self.l1 = Conv(int(64 * w), int(128 * w), 3, 2, 1)
        self.l3 = Conv(int(128 * w), int(256 * w), 3, 2, 1)
        self.l5 = Conv(int(256 * w), int(512 * w), 3, 2, 1)
        self.l7 = Conv(int(512 * w), int(512 * w * r), 3, 2, 1)

        self.l2 = C2f(int(128 * w), int(128 * w), n, True)
        self.l4 = C2f(int(256 * w), int(256 * w), 2 * n, True)
        self.l6 = C2f(int(512 * w), int(512 * w), 2 * n, True)
        self.l8 = C2f(int(512 * w * r), int(512 * w * r), n, True)

        self.l9 = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """

        # Stem Layer
        stemLayer = self.l0(x)
        
        # Stage Layer 1
        feat0 = self.l1(stemLayer)
        feat0 = self.l2(feat0)

        # Stage Layer 2
        feat1 = self.l3(feat0)
        feat1 = self.l4(feat1)

        # Stage Layer 3
        feat2 = self.l5(feat1)
        feat2 = self.l6(feat2)

        # Stage Layer 4
        feat3 = self.l7(feat2)
        feat3 = self.l8(feat3)
        feat3 = self.l9(feat3)

        return feat0, feat1, feat2, feat3
