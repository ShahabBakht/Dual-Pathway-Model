import sys

import torch
import torch.nn as nn

sys.path.append('../backbone/')
from monkeynet import VisualNet

class VisualNet_classifier(nn.Module):
    def __init__(self, num_classes, num_res_blocks = 10, num_paths = 1):
        super(VisualNet_classifier).__init__()

        self.visualnet = VisualNet(num_res_blocks = 10)
        self.linear = nn.linear(self.visualnet.path1.resblocks_out_channels,num_classes)

    def forward(self, x):

        features = self.visualnet(x)
        y = self.linear(features)

        return y


