"""
Author: Adapted for 3D by ChatGPT (from Omid Nejati's MedViT2D)
Email: omid_nejaty@alumni.iust.ac.ir

MedViT-3D: A Robust Vision Transformer for Generalized Medical Image Classification (Volumetric).
"""

from functools import partial
import math
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from utils import merge_pre_bn

NORM_EPS = 1e-5


# -----------------------------
# Basic 3D building blocks
# -----------------------------

class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        # padding=1 assumes kernel_size=3; it also works for 1 with no effect (harmless)
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=1, groups=groups, bias=False
        )
        self.norm = nn.BatchNorm3d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool3d((2, 2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA3D(nn.Module):
    """
    Multi-Head Convolutional Attention (3D)
    """
    def __init__(self, out_channels, head_dim):
        super().__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        assert out_channels % head_dim == 0, "out_channels must be divisible by head_dim for grouped conv"
        self.group_conv3x3 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            groups=out_channels // head_dim, bias=False
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer3D(nn.Module):
    """
    ECA adapted to 3D: global average over D,H,W then channel-wise 1D conv
    """
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super().__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid() if sigmoid else h_sigmoid()

    def forward(self, x):
        # x: (B,C,D,H,W) -> (B,C,1,1,1)
        y = self.avg_pool(x)                     # (B,C,1,1,1)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))  # (B,1,C) conv over channel
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)         # (B,C,1,1,1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class LocalityFeedForward3D(nn.Module):
    """
    3D version of LocalityFeedForward (Conv-DepthwiseConv-Conv with optional SE/ECA)
    """
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # First "linear" -> 1x1x1 conv
        layers.extend([
            nn.Conv3d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(hidden_dim, eps=NORM_EPS),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
        ])

        # Depth-wise conv (3x3x3)
        if not wo_dp_conv:
            dp = [
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim, eps=NORM_EPS),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer3D(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer3D(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError(f'Activation type {attn} is not implemented')

        # Second "linear" -> 1x1x1 conv
        layers.extend([
            nn.Conv3d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_dim, eps=NORM_EPS)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


class Mlp3D(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv3d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        # kept for API parity; can be a no-op if unused
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


# -----------------------------
# Core Blocks (ECB, E-MHSA, LTB)
# -----------------------------

class ECB3D(nn.Module):
    """
    Efficient Convolution Block (3D)
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed3D(in_channels, out_channels, stride)
        self.mhca = MHCA3D(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.conv = LocalityFeedForward3D(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

        self.norm = norm_layer(out_channels)
        self.is_bn_merged = False  # API compatibility

    def merge_bn(self):
        # no-op kept for API compatibility
        self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.conv(out)
        return x


class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention over tokens (sequence). This part remains mostly the same,
    but the spatial reduction uses sr_ratio**3 for 3D volumes.
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # 3D tokens => reduction multiplies over D,H,W
        self.N_ratio = sr_ratio ** 3
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)  # (B, C, N)
            x_ = self.sr(x_)       # downsample tokens
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LTB3D(nn.Module):
    """
    Local Transformer Block (3D)
    Mix of MHSA (token) and MHCA (conv) paths, then local feed-forward.
    """
    def __init__(
        self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
        mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm3d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed3D(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(
            self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed3D(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA3D(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward3D(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, D, H, W = x.shape

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x

        # tokens: (B, N, C) where N = D*H*W
        out_tokens = rearrange(out, "b c d h w -> b (d h w) c")
        out_tokens = self.mhsa_path_dropout(self.e_mhsa(out_tokens))
        x = x + rearrange(out_tokens, "b (d h w) c -> b c d h w", d=D, h=H, w=W)

        out2 = self.projection(x)
        out2 = out2 + self.mhca_path_dropout(self.mhca(out2))
        x = torch.cat([x, out2], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out3 = self.norm2(x)
        else:
            out3 = x

        x = x + self.conv(out3)
        return x


# -----------------------------
# MedViT-3D Model
# -----------------------------

class MedViT3D(nn.Module):
    def __init__(
        self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000, 
        strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
        use_checkpoint=False, in_chans = 1
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.stage_out_channels = [
            [96] * (depths[0]),
            [192] * (depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (depths[2] // 5),
            [768] * (depths[3] - 1) + [1024]
        ]

        # Next Hybrid Strategy
        self.stage_block_types = [
            [ECB3D] * depths[0],
            [ECB3D] * (depths[1] - 1) + [LTB3D],
            [ECB3D, ECB3D, ECB3D, ECB3D, LTB3D] * (depths[2] // 5),
            [ECB3D] * (depths[3] - 1) + [LTB3D]
        ]

        # Stem: two convs, then a stride-2 conv to downsample, replicated from 2D design
        self.stem = nn.Sequential(
            ConvBNReLU3D(in_chans, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU3D(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU3D(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU3D(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )

        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule

        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                stride = 2 if (strides[stage_id] == 2 and block_id == 0) else 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]

                if block_type is ECB3D:
                    layer = ECB3D(
                        input_channel, output_channel,
                        stride=stride, path_dropout=dpr[idx + block_id],
                        drop=drop, head_dim=head_dim
                    )
                    features.append(layer)
                elif block_type is LTB3D:
                    layer = LTB3D(
                        input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                        sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                        attn_drop=attn_drop, drop=drop
                    )
                    features.append(layer)

                input_channel = output_channel
            idx += numrepeat

        self.features = nn.Sequential(*features)
        self.norm = nn.BatchNorm3d(output_channel, eps=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        print('initialize_weights...')
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for _, module in self.named_modules():
            if isinstance(module, (ECB3D, LTB3D)):
                module.merge_bn()

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d,)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, D, H, W)
        x = self.stem(x)
        for layer in self.features:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)               # (B, C, 1, 1, 1)
        x = torch.flatten(x, 1)           # (B, C)
        x = self.proj_head(x)             # (B, num_classes)
        return x


# -----------------------------
# Factory functions
# -----------------------------

@register_model
def MedViT3D_small(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedViT3D(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, in_chans=1, **kwargs)
    return model


@register_model
def MedViT3D_base(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedViT3D(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, in_chans = 1, **kwargs)
    return model


@register_model
class MedViT3D_large(nn.Module):
    def __new__(cls, *args, **kwargs):
        # keep same API pattern as register_model funcs
        return MedViT3D(stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2,in_chans=1, **kwargs)
