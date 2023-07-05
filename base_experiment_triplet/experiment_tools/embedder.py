import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, downsample=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18Embedder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, kernel_size=3):
        super(ResNet18Embedder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            out_channels, out_channels, kernel_size, 2, stride=1, padding=1
        )
        self.layer2 = self._make_layer(
            out_channels, out_channels * 2, kernel_size, 2, stride=2, padding=1
        )
        self.layer3 = self._make_layer(
            out_channels * 2, out_channels * 4, kernel_size, 2, stride=2, padding=1
        )
        self.layer4 = self._make_layer(
            out_channels * 4, out_channels * 8, kernel_size, 2, stride=2, padding=1
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, kernel_size, num_blocks, stride, padding):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(
            BasicBlock(in_channels, out_channels, kernel_size, stride, padding, downsample)
        )
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, kernel_size, padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def conv_dw(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels, in_channels, kernel_size, stride, groups=in_channels),
        nn.BatchNorm1d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetV3Embedder(nn.Module):
    def __init__(self, in_channels=1):
        super(MobileNetV3Embedder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            conv_dw(16, 64, 3, 2),
            conv_dw(64, 128, 3, 2),
            conv_dw(128, 128, 3, 1),
            conv_dw(128, 256, 3, 2),
            conv_dw(256, 256, 3, 1),
            conv_dw(256, 512, 3, 2),
            conv_dw(512, 512, 3, 1),
            conv_dw(512, 512, 3, 1),
            conv_dw(512, 512, 3, 1),
            conv_dw(512, 512, 3, 1),
            conv_dw(512, 1024, 3, 2),
            conv_dw(1024, 1024, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x
