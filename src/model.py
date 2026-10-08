"""FruitLite-L: compact residual CNN; identical graph for 224 and 32 inputs."""

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.layers(self.pool(x))


class FruitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=2, multiscale=True, attention=True):
        super().__init__()
        hidden = in_channels * expansion
        self.expand = ConvBNAct(in_channels, hidden)
        self.multiscale = multiscale
        self.local = ConvBNAct(hidden // 2 if multiscale else hidden, hidden // 2 if multiscale else hidden, 3, stride, hidden // 2 if multiscale else hidden)
        self.context = ConvBNAct(hidden // 2, hidden // 2, 5, stride, hidden // 2) if multiscale else None
        self.gate = ChannelGate(hidden) if attention else nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        expanded = self.expand(x)
        if self.multiscale:
            left, right = expanded.chunk(2, dim=1)
            features = torch.cat((self.local(left), self.context(right)), dim=1)
        else:
            features = self.local(expanded)
        return self.activation(self.project(self.gate(features)) + self.shortcut(x))


class FruitLite(nn.Module):
    def __init__(self, num_classes=5, multiscale=True, attention=True, dropout=0.2):
        super().__init__()
        self.stem = ConvBNAct(3, 16, 3, 2)
        stages = []
        channels = 16
        for output, repeat, first_stride in ((32, 1, 1), (56, 2, 2), (96, 3, 2), (160, 2, 2)):
            for index in range(repeat):
                stages.append(FruitBlock(channels, output, first_stride if index == 0 else 1,
                                         multiscale=multiscale, attention=attention and output >= 96))
                channels = output
        self.stages = nn.Sequential(*stages)
        self.head = ConvBNAct(channels, 224)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(224, num_classes))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.head(self.stages(self.stem(x)))).flatten(1))