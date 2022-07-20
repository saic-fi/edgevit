# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

layer_scale = False
init_value = 1e-6

class AfterReconstruction(nn.Identity):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):

        N = tensor.shape[-2]

        '''grid samplig'''
        if N == 56*56:
            # indices = [0, 8, 16, 24, 32, 40, 48, 448, 456, 464, 472, 480,
            #            488, 496, 896, 904, 912, 920, 928, 936, 944, 1344, 1352, 1360,
            #            1368, 1376, 1384, 1392, 1792, 1800, 1808, 1816, 1824, 1832, 1840, 2240,
            #            2248, 2256, 2264, 2272, 2280, 2288, 2688, 2696, 2704, 2712, 2720, 2728,
            #            2736]
            indices = [   0,    4,    8,   12,   16,   20,   24,   28,   32,   36,   40,   44,
                          48,   52,  224,  228,  232,  236,  240,  244,  248,  252,  256,  260,
                          264,  268,  272,  276,  448,  452,  456,  460,  464,  468,  472,  476,
                          480,  484,  488,  492,  496,  500,  672,  676,  680,  684,  688,  692,
                          696,  700,  704,  708,  712,  716,  720,  724,  896,  900,  904,  908,
                          912,  916,  920,  924,  928,  932,  936,  940,  944,  948, 1120, 1124,
                          1128, 1132, 1136, 1140, 1144, 1148, 1152, 1156, 1160, 1164, 1168, 1172,
                          1344, 1348, 1352, 1356, 1360, 1364, 1368, 1372, 1376, 1380, 1384, 1388,
                          1392, 1396, 1568, 1572, 1576, 1580, 1584, 1588, 1592, 1596, 1600, 1604,
                          1608, 1612, 1616, 1620, 1792, 1796, 1800, 1804, 1808, 1812, 1816, 1820,
                          1824, 1828, 1832, 1836, 1840, 1844, 2016, 2020, 2024, 2028, 2032, 2036,
                          2040, 2044, 2048, 2052, 2056, 2060, 2064, 2068, 2240, 2244, 2248, 2252,
                          2256, 2260, 2264, 2268, 2272, 2276, 2280, 2284, 2288, 2292, 2464, 2468,
                          2472, 2476, 2480, 2484, 2488, 2492, 2496, 2500, 2504, 2508, 2512, 2516,
                          2688, 2692, 2696, 2700, 2704, 2708, 2712, 2716, 2720, 2724, 2728, 2732,
                          2736, 2740, 2912, 2916, 2920, 2924, 2928, 2932, 2936, 2940, 2944, 2948,
                          2952, 2956, 2960, 2964]
            # tensor = tensor[:, :, :, indices, :]
            tensor = tensor[:, indices, :]
        elif N == 28*28:
            # indices = [0, 4, 8, 12, 16, 20, 24, 112, 116, 120, 124, 128, 132, 136,
            #            224, 228, 232, 236, 240, 244, 248, 336, 340, 344, 348, 352, 356, 360,
            #            448, 452, 456, 460, 464, 468, 472, 560, 564, 568, 572, 576, 580, 584,
            #            672, 676, 680, 684, 688, 692, 696]
            indices = [  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
                         56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,  80,  82,
                         112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
                         168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194,
                         224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250,
                         280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306,
                         336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362,
                         392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418,
                         448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474,
                         504, 506, 508, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530,
                         560, 562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586,
                         616, 618, 620, 622, 624, 626, 628, 630, 632, 634, 636, 638, 640, 642,
                         672, 674, 676, 678, 680, 682, 684, 686, 688, 690, 692, 694, 696, 698,
                         728, 730, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752, 754]
            # tensor = tensor[:, :, :, indices, :]
            tensor = tensor[:, indices, :]
        elif N ==14*14:
            indices = [0, 2, 4, 6, 8, 10, 12, 28, 30, 32, 34, 36, 38, 40,
                       56, 58, 60, 62, 64, 66, 68, 84, 86, 88, 90, 92, 94, 96,
                       112, 114, 116, 118, 120, 122, 124, 140, 142, 144, 146, 148, 150, 152,
                       168, 170, 172, 174, 176, 178, 180]
            # tensor = tensor[:, :, :, indices, :]
            tensor = tensor[:, indices, :]

        return tensor

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Attention(nn.Module):
#     """
#     GSA: using a  key to summarize the information for a group to be efficient.
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.sampler = AfterReconstruction()
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         # x_ = x.reshape(B, N, C)
#         x_ = self.sampler(x)
#         kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#         # q = qkv[0]
#         # kv = self.sampler(qkv[1:])
#         # k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.sampler = AfterReconstruction()
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = self.sampler(x)
#         qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
#         if H == 14:
#             x = x.permute(0, 2, 1).reshape(B, C, 7, 7)
#             x = nn.Upsample(scale_factor=2, mode='nearest')(x)
#             x = x.reshape(B, C, -1).permute(0, 2, 1)
#         elif H == 56 or H == 28:
#             x = x.permute(0, 2, 1).reshape(B, C, 14, 14)
#             x = nn.Upsample(scale_factor=H//14, mode='nearest')(x)
#             x = x.reshape(B, C, -1).permute(0, 2, 1)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class Attention_AvgSR(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr = sr_ratio

        if self.sr > 1:
            kernel_size = sr_ratio
            self.upsample= nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)

            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr_ = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H:int, W:int):
        B, N, C = x.shape
        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr_(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H/self.sr), int(W/self.sr))
            x = self.upsample(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sampler = AfterReconstruction()
        # self.upsample = nn.Upsample(scale_factor=sr_ratio, mode='nearest')
        self.sr= sr_ratio
        # if self.sr > 1:
        kernel_size = sr_ratio
        self.upsample= nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
        self.norm = nn.LayerNorm(dim)

        self.pool = nn.AdaptiveMaxPool2d(7)
        self.sr_ = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, H:int, W:int):
        B, N, C = x.shape
        # x = self.sampler(x)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr_(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H/self.sr), int(W/self.sr))
            x = self.upsample(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1., SA=Attention):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = SA(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # global layer_scale
        # self.ls = layer_scale
        # if self.ls:
        #     global init_value
        #     print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
        #     self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
        #     self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # if self.ls:
        #     x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        #     x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        # else:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class CSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1., SA=Attention):
        super().__init__()
        self.cblock = CBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                             drop_path, act_layer, norm_layer)
        self.sablock = SABlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                               drop_path, act_layer, norm_layer, sr_ratio, SA=SA)

    def forward(self, x):
        x = self.cblock(x)
        x = self.sablock(x)
        return x

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
    
class UniFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, conv_stem=False, sr_ratios=[8,4,2,1],
                 SA=Attention):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        if conv_stem:
            self.patch_embed1 = head_embedding(in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2])
            self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3])
        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CSABlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], SA=SA)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CSABlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer, sr_ratio=sr_ratios[1], SA=SA)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            CSABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, sr_ratio=sr_ratios[2], SA=SA)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
        for i in range(depth[3])])
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x


@register_model
def uniformer_twins_05G_MaxSR_deconv(pretrained=True, **kwargs):
    model = UniFormer(
        depth=[1, 1, 3, 2],
        embed_dim=[36, 72, 144, 288], head_dim=36, mlp_ratio=[4]*4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratios=[8,4,2,1], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def uniformer_twins_05G_AvgSR_deconv(pretrained=True, **kwargs):
    model = UniFormer(
        depth=[1, 1, 3, 2],
        embed_dim=[36, 72, 144, 288], head_dim=36, mlp_ratio=[4]*4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratios=[8,4,2,1], SA=Attention_AvgSR, **kwargs)
    model.default_cfg = _cfg()
    return model