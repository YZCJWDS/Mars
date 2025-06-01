import torch
import torch.nn as nn
from model.base.components import Conv, C2f


class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    YOLOv8 Neck with FPN + PAN structure
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        # Channel configurations
        c2 = int(256 * w)  # feat1 channels
        c3 = int(512 * w)  # feat2 channels  
        c4 = int(512 * w * r)  # feat3 channels

        # FPN layers (top-down pathway)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_conv1 = Conv(c4, c3, 1, 1)  # reduce channels for feat3
        self.c2f_fpn1 = C2f(c3 + c3, c3, n)  # concat feat2 + upsampled feat3
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_conv2 = Conv(c3, c2, 1, 1)  # reduce channels for merged feat2
        self.c2f_fpn2 = C2f(c2 + c2, c2, n)  # concat feat1 + upsampled merged feat2

        # PAN layers (bottom-up pathway)
        self.downsample1 = Conv(c2, c2, 3, 2)  # downsample P3 to P4 size
        self.c2f_pan1 = C2f(c2 + c3, c3, n)  # concat with FPN P4
        
        self.downsample2 = Conv(c3, c3, 3, 2)  # downsample P4 to P5 size
        self.c2f_pan2 = C2f(c3 + c4, c4, n)  # concat with original feat3

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        # FPN top-down pathway
        # P5 -> P4
        p5_reduced = self.reduce_conv1(feat3)  # (B, 512*w, 20, 20)
        p5_up = self.upsample1(p5_reduced)  # (B, 512*w, 40, 40)
        p4_concat = torch.cat([feat2, p5_up], dim=1)  # (B, 1024*w, 40, 40)
        p4_out = self.c2f_fpn1(p4_concat)  # (B, 512*w, 40, 40)
        
        # P4 -> P3
        p4_reduced = self.reduce_conv2(p4_out)  # (B, 256*w, 40, 40)
        p4_up = self.upsample2(p4_reduced)  # (B, 256*w, 80, 80)
        p3_concat = torch.cat([feat1, p4_up], dim=1)  # (B, 512*w, 80, 80)
        p3_out = self.c2f_fpn2(p3_concat)  # (B, 256*w, 80, 80)

        # PAN bottom-up pathway
        # P3 -> P4
        p3_down = self.downsample1(p3_out)  # (B, 256*w, 40, 40)
        p4_pan_concat = torch.cat([p3_down, p4_out], dim=1)  # (B, 768*w, 40, 40)
        p4_final = self.c2f_pan1(p4_pan_concat)  # (B, 512*w, 40, 40)
        
        # P4 -> P5
        p4_down = self.downsample2(p4_final)  # (B, 512*w, 20, 20)
        p5_pan_concat = torch.cat([p4_down, feat3], dim=1)  # (B, 1024*w*r, 20, 20)
        p5_final = self.c2f_pan2(p5_pan_concat)  # (B, 512*w*r, 20, 20)

        # Return in the expected format: (C, X, Y, Z)
        C = p4_final  # (B, 512*w, 40, 40)
        X = p3_out    # (B, 256*w, 80, 80)
        Y = p4_final  # (B, 512*w, 40, 40)
        Z = p5_final  # (B, 512*w*r, 20, 20)
        
        return C, X, Y, Z