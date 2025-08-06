import torch
import torch.nn as nn

class Conv3DBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride == 2:
            self.pool = nn.AvgPool3d(2, stride=2, ceil_mode=True)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.norm = nn.BatchNorm3d(out_channels)
        elif in_channels != out_channels:
            self.pool = nn.Identity()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.pool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        ca = self.sigmoid(avg + max_)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * sa

class LTB_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, out_channels, stride)
        self.cbam = CBAM3D(out_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cbam(x)
        x = self.conv(x)
        return x

class MedViT3D(nn.Module):
    def __init__(self, stem_channels, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3DBNReLU(1, stem_channels[0], kernel_size=3, stride=2),
            Conv3DBNReLU(stem_channels[0], stem_channels[1], kernel_size=3, stride=1),
            Conv3DBNReLU(stem_channels[1], stem_channels[2], kernel_size=3, stride=1),
            Conv3DBNReLU(stem_channels[2], stem_channels[2], kernel_size=3, stride=2)
        )

        self.ltb1 = LTB_CBAM(stem_channels[2], 64)
        self.ltb2 = LTB_CBAM(64, 128)
        self.ltb3 = LTB_CBAM(128, 256)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W] -> [B, C, D, H, W]
        x = self.stem(x)
        x = self.ltb1(x)
        x = self.ltb2(x)
        x = self.ltb3(x)
        x = self.avgpool(x).flatten(1)
        x = self.head(x)
        return x

# Optional: register_model if using timm or custom registry
from timm.models.registry import register_model

@register_model
def MedViT3D_small(pretrained=False, **kwargs):
    model = MedViT3D(stem_channels=[32, 64, 128], num_classes=kwargs.get('num_classes', 4))
    return model
