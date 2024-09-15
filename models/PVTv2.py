import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

import math


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)

def conv3x3(in_planes, out_planes, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
              bias=False, weight_std=False):

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads.cuda()], dim=1))

    return out


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False,
                               weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out

class BasicBlock_2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1), downsample=None, weight_std=False):
        super(BasicBlock_2D, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_std=weight_std)
        self.norm1 = nn.InstanceNorm2d(planes, affine=True)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=3, stride=(1, 1), padding=1, bias=False,
                               weight_std=weight_std)
        self.norm2 = nn.InstanceNorm2d(planes, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.conv3 = conv3x3x3(planes, planes * 4, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm3 = Norm_layer(norm_cfg, planes * 4)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.nonlin(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr_2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool_2D = nn.AdaptiveAvgPool2d(7)
            self.pool = nn.AdaptiveAvgPool3d(7)
            self.sr = nn.Conv3d(dim, dim, kernel_size=1, stride=1)
            self.sr_2D = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
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
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def return_attention_map(self):
        return self.attn

    def forward(self, x, D, H, W):
        if D != -1:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
                x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            self.attn = self.attn_drop(attn)

            x = (self.attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                    x_ = self.sr_2D(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr_2D(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def return_attention_map(self):
        return self.attn.return_attention_map()

    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x

class OverlapPatchEmbed_first(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[16, 96, 96], patch_size=[8, 8, 8], stride=[2,2,2], in_chans=[1,1], embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.D, self.H, self.W = img_size[0] // stride[0], img_size[1] // stride[1], img_size[2] // stride[2]
        self.num_patches = self.H * self.W * self.D
        stem = [nn.Conv3d(in_chans[0], embed_dim, kernel_size=patch_size, stride=[2,2,2],
                padding = (patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)),
                     nn.InstanceNorm3d(embed_dim),
                     nn.LeakyReLU(negative_slope=1e-2, inplace=True)
                     ]
        for i in range(2):
            stem.append(nn.Conv3d(embed_dim, embed_dim, 3, 1, padding=1, bias=False))
            stem.append(nn.InstanceNorm3d(embed_dim))
            stem.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.proj = nn.Sequential(*stem)
        self.proj_next = nn.Conv3d(embed_dim, embed_dim, kernel_size=[3,3,3], stride=[1,2,2], padding=(1,1,1))

        stem_2D = [nn.Conv2d(in_chans[1], embed_dim, kernel_size=(patch_size[1], patch_size[2]), stride=[2, 2],
                          padding=(patch_size[1] // 2, patch_size[2] // 2)),
                nn.InstanceNorm2d(embed_dim),
                nn.LeakyReLU(negative_slope=1e-2, inplace=True)
                ]
        for i in range(2):
            stem_2D.append(nn.Conv2d(embed_dim, embed_dim, 3, 1, padding=1, bias=False))
            stem_2D.append(nn.InstanceNorm2d(embed_dim))
            stem_2D.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.proj_2D = nn.Sequential(*stem_2D)
        self.proj_next_2D = nn.Conv2d(embed_dim, embed_dim, kernel_size=[3, 3], stride=[2, 2], padding=(1, 1))


        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        if len(x.shape) == 5:
            x_1 = self.proj(x)
            x = self.proj_next(x_1)
            _, _, D, H, W = x.shape
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.norm(x)
            return (x_1, x), D, H, W
        else:
            x_1 = self.proj_2D(x)
            x = self.proj_next_2D(x_1)
            _, _, H, W = x.shape
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.norm(x)
            return (x_1, x), H, W


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[16, 96, 96], patch_size=[8, 8, 8], stride=4, in_chans=1, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.D, self.H, self.W = img_size[0] // stride[0], img_size[1] // stride[1], img_size[2] // stride[2]
        self.num_patches = self.H * self.W * self.D
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.proj_2D = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size[1], patch_size[2]), stride=(stride[1], stride[2]),
                              padding=(patch_size[1] // 2, patch_size[2] // 2))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) == 5:
            x = self.proj(x)
            _, _, D, H, W = x.shape
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.norm(x)
            return x, D, H, W
        else:
            x = self.proj_2D(x)
            _, _, H, W = x.shape
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.norm(x)
            return x, H, W

class OverlapPatchEmbed_decoder(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[96, 96], patch_size=4, stride=4, in_chans=1, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_2D = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=(patch_size,patch_size), stride=(stride, stride), bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.proj_2D(x)
        _, _, H, W = x.shape
        x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=[16, 96, 96], patch_size=16, in_chans=[1,1], num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[4, 2, 1, 1], num_stages=4, linear=False):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.MODEL_NUM_CLASSES = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = OverlapPatchEmbed_first(
                    img_size=img_size if i == 0 else [sub_size // (2 ** (i + 1)) for sub_size in img_size],
                    patch_size=[7, 7, 7],
                    stride=[2,4,4] if i == 0 else [2, 2, 2],
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(
                    img_size=img_size if i == 0 else [sub_size // (2 ** (i + 1)) for sub_size in img_size],
                    patch_size=[7, 7, 7] if i == 0 else [3, 3, 3],
                    stride=[2, 2, 2] if i == 0 else [2, 2, 2],
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, (Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if len(x.shape) == 5:
            B = x.shape[0]
            outs = []
            for i in range(self.num_stages):
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                block = getattr(self, f"block{i + 1}")
                norm = getattr(self, f"norm{i + 1}")
                if i == 0:
                    (x_1, x), D, H, W = patch_embed(x)
                else:
                    x, D, H, W = patch_embed(x)
                for blk in block:
                    x = blk(x, D, H, W)
                x = norm(x)
                x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(x)

            return outs
        else:
            B = x.shape[0]
            outs = []

            for i in range(self.num_stages):

                patch_embed = getattr(self, f"patch_embed{i + 1}")
                block = getattr(self, f"block{i + 1}")
                norm = getattr(self, f"norm{i + 1}")
                if i == 0:
                    (x_1, x), H, W = patch_embed(x)
                else:
                    x, H, W = patch_embed(x)
                for blk in block:
                    x = blk(x, -1, H, W)
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x)

            return outs

    def forward(self, x):
        outs = self.forward_features(x)
        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict



class pvt_v2_b1(PyramidVisionTransformerV2):
    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=2, weight_std=False):
        print("student")
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.gap_2D = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        final_outs = []
        outs = self.forward_features(x)
        if len(outs[-1].size()) == 5:
            for out in outs:
                final_outs.append(self.gap(out).flatten(1))
            return final_outs
        else:
            for out in outs:
                final_outs.append(self.gap_2D(out).flatten(1))
            return final_outs

class pvt_v2_b1_tea(PyramidVisionTransformerV2):
    def __init__(self):
        print("teacher")
        super(pvt_v2_b1_tea, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.gap_2D = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        outs = self.forward_features(x)
        if len(outs[-1].size()) == 5:
            return self.gap(outs[-1]).flatten(1)
        else:
            return self.gap_2D(outs[-1]).flatten(1)
