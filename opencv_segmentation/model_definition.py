import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        layers.append(ConvBNReLU(inp, int(inp * expand_ratio), kernel_size=1))
        layers.extend([
            ConvBNReLU(int(inp * expand_ratio), int(inp * expand_ratio), stride=stride, groups=int(inp * expand_ratio)),
            nn.Conv2d(int(inp * expand_ratio), oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class DeepLabV3_MobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_MobileNetV3, self).__init__()
        self.backbone = nn.Sequential(
            ConvBNReLU(3, 16, stride=2),
            InvertedResidual(16, 24, stride=2, expand_ratio=4),
            InvertedResidual(24, 24, stride=1, expand_ratio=3),
            InvertedResidual(24, 40, stride=2, expand_ratio=3),
            InvertedResidual(40, 40, stride=1, expand_ratio=3),
            InvertedResidual(40, 80, stride=2, expand_ratio=6),
            InvertedResidual(80, 80, stride=1, expand_ratio=2.5),
            InvertedResidual(80, 112, stride=1, expand_ratio=6),
            InvertedResidual(112, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x


def load_deeplab_model(weights_path='deeplabv3_mobilenet_v3_large.pth'):
    model = DeepLabV3_MobileNetV3(num_classes=21)
    state_dict = torch.load(weights_path, map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "aux_classifier" not in k}
    # 여기서 size mismatch 문제를 해결합니다.
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model
