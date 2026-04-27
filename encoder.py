import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=None)

        # Modify first layer for 1 channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract layers
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # (64, H/2)

        self.pool = resnet.maxpool  # (64, H/4)

        self.layer1 = resnet.layer1  # (64)
        self.layer2 = resnet.layer2  # (128)
        self.layer3 = resnet.layer3  # (256)
        self.layer4 = resnet.layer4  # (512)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4