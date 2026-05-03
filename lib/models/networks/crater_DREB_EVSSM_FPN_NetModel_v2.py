# -*- coding: utf-8 -*-
"""
Version2 的增强版本
模型网络的最终版本
检测分支为CenterNet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels import inceptionresnetv2
import numpy as np
import numbers
import math
from einops import rearrange, repeat

# mamba selective scan，使用Mamba的可选择机制进行扫描，得出的分析。
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

##########################################
# ======== 基础模块 定义卷积操作和BN操作在一个小块一起执行 ========
##########################################

"""
新添加的核心代码图像，用于保存图像
"""

import os
import matplotlib.pyplot as plt

def save_feature_map(tensor, name, save_dir="debug_feats"):
    """
    tensor: BCHW
    只保存每个特征图的通道平均结果（避免1000+通道）
    """
    os.makedirs(save_dir, exist_ok=True)

    # → 转成 CPU numpy [H, W]
    fmap = tensor.detach().cpu().mean(1).squeeze(0).numpy()

    plt.figure(figsize=(4,4))
    plt.imshow(fmap, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, 1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, 1, 1, bias=True)

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = F.relu(self.down(x))
        x = torch.sigmoid(self.up(x))
        return inputs * x.view(-1, inputs.size(1), 1, 1)


class RepVGGBlock_useSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.se = SEBlock(out_channels, max(out_channels // 16, 1))
        self.nonlinearity = nn.ReLU(inplace=True)
        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels, out_channels, 3, stride, 1)
        self.rbr_1x1 = conv_bn(in_channels, out_channels, 1, stride, 0)

    def forward(self, inputs):
        id_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0
        out = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        return self.nonlinearity(self.se(out))


##########################################
# ======== 特征融合模块 MAGFF ========
##########################################
class MAGFF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(MAGFF, self).__init__()
        inter = channels // r
        self.local = nn.Sequential(
            nn.Conv2d(channels, inter, 1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1),
            nn.BatchNorm2d(channels),
        )
        self.globalp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x_sum = x1 + x2
        weight = self.sigmoid(self.local(x_sum) + self.globalp(x_sum))
        return x1 * weight + x2 * (1 - weight)


##########################################
# ======== 空域自适应高斯增强模块（改进版） ========
##########################################
class AdaptiveGaussianEnhance_v2(nn.Module):
    def __init__(self, channels, kernel_size=5, init_sigma=1.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.alpha = nn.Parameter(torch.ones(channels, 1, 1))  # 每通道可学习增强权重
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))

        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        self.register_buffer("xx", xx.clone())
        self.register_buffer("yy", yy.clone())

        # channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def gaussian_kernel(self, sigma):
        kernel = torch.exp(-(self.xx ** 2 + self.yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        return kernel

    def forward(self, x):
        sigma = F.softplus(self.log_sigma) + 1e-6
        kernel = self.gaussian_kernel(sigma)
        blur = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=self.channels)
        sharp = x - blur
        out = x + self.alpha * sharp
        out = out * self.ca(out)
        return out


##########################################
# ======== S6 注意力模块 ========
##########################################
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=3, expand=2., dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.GELU()

        self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2), out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3, groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        return out_y[:, 0]

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        y = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out


class S6GatingAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.s6 = SS2D(d_model=channels)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        proj = self.pre_proj(x)
        s6_out = self.s6(proj)
        gate = self.sigmoid(self.gate_conv(s6_out))
        return x * gate + x


##########################################
# ======== 改进版 FPN + FPNInception ========
##########################################
class FPN_SSAttn_v2(nn.Module):
    def __init__(self, norm_layer, num_filters=256):
        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(self.inception.conv2d_2a, self.inception.conv2d_2b, self.inception.maxpool_3a)
        self.enc2 = nn.Sequential(self.inception.conv2d_3b, self.inception.conv2d_4a, self.inception.maxpool_5a)
        self.enc3 = nn.Sequential(self.inception.mixed_5b, self.inception.repeat, self.inception.mixed_6a)
        self.enc4 = nn.Sequential(self.inception.repeat_1, self.inception.mixed_7a)

        self.s6_0 = S6GatingAttention(32)
        self.s6_1 = S6GatingAttention(64)
        self.s6_2 = S6GatingAttention(192)
        self.s6_3 = S6GatingAttention(1088)

        self.lateral4 = nn.Conv2d(2080, num_filters, 1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, 1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, 1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, 1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, 1, bias=False)

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))

    def forward(self, x):
        enc0 = self.s6_0(self.enc0(x))
        enc1 = self.s6_1(self.enc1(enc0))
        enc2 = self.s6_2(self.enc2(enc1))
        enc3 = self.s6_3(self.enc3(enc2))
        enc4 = self.enc4(enc3)

        lat4 = self.lateral4(enc4)
        lat3 = self.lateral3(enc3)
        lat2 = self.lateral2(enc2)
        lat1 = self.lateral1(enc1)
        lat0 = self.lateral0(enc0)

        map4 = lat4
        map3 = self.td1(lat3 + F.interpolate(map4, size=lat3.shape[-2:], mode='bilinear', align_corners=False))
        map2 = self.td2(lat2 + F.interpolate(map3, size=lat2.shape[-2:], mode='bilinear', align_corners=False))
        map1 = self.td3(lat1 + F.interpolate(map2, size=lat1.shape[-2:], mode='bilinear', align_corners=False))

        return lat0, map1, map2, map3, map4


class FPNInception_SSAttn_v2(nn.Module):
    def __init__(self, norm_layer, num_filters=128, num_filters_fpn=256):
        super().__init__()
        self.fpn = FPN_SSAttn_v2(norm_layer, num_filters_fpn)
        self.smooth = nn.Sequential(nn.Conv2d(num_filters_fpn * 4, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU())
        self.final = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 在 FPNInception_SSAttn_v2.forward 中修改
        map0, map1, map2, map3, map4 = self.fpn(x)
        
        # 将所有 FPN 高层上采样到 map1 大小
        target_size = map1.shape[-2:]
        up_feats = [F.interpolate(m, size=target_size, mode='bilinear', align_corners=False) for m in [map1, map2, map3, map4]]
        concat_feats = torch.cat(up_feats, dim=1)
        smoothed = self.smooth(concat_feats)
        
        # 将 smoothed 上采样到 map0 大小再融合
        smoothed = F.interpolate(smoothed, size=map0.shape[-2:], mode='bilinear', align_corners=False)
        smoothed = smoothed + map0  # 尺寸一致，避免报错
        
        smoothed2 = self.smooth2(smoothed)
        up_feat = F.interpolate(smoothed2, size=x.shape[2:], mode='bilinear', align_corners=False)
        final = torch.tanh(self.final(up_feat)) + x
        final = torch.clamp(final, -1, 1)
        return smoothed2, final


##########################################
# ======== 主干 DREB_Net 整合版 ========
##########################################
class DREB_FPN_Net_SSAttn_v2(nn.Module):
    def __init__(self, heads=None, head_conv=64):
        super().__init__()
        self.heads = heads
        self.in_planes = 64

        self.stage0 = RepVGGBlock_useSE(3, 64, 3, stride=2)
        self.stage1 = self._make_stage(64, 3, stride=2)
        self.stage2 = self._make_stage(128, 5, stride=2)
        self.stage3 = self._make_stage(256, 8, stride=2)

        self.deconv_layers1 = self._make_deconv_layer(256, 4)
        self.deconv_layers2 = self._make_deconv_layer(256, 4)

        self.deblur_FPN = FPNInception_SSAttn_v2(norm_layer=nn.BatchNorm2d, num_filters=128, num_filters_fpn=256)

        self.feature_fusion = MAGFF(channels=128)
        self.AGE = AdaptiveGaussianEnhance_v2(channels=128, kernel_size=5, init_sigma=1.5)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(256, head_conv, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 1)
            )
            self.__setattr__(head, fc)

    def _make_stage(self, planes, num_blocks, stride):
        blocks = [RepVGGBlock_useSE(self.in_planes, planes, 3, stride=stride)]
        self.in_planes = planes
        for _ in range(num_blocks - 1):
            blocks.append(RepVGGBlock_useSE(planes, planes, 3))
        return nn.ModuleList(blocks)

    def _make_deconv_layer(self, planes, kernel):
        return nn.Sequential(nn.ConvTranspose2d(self.in_planes, planes, kernel_size=kernel, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(planes),
                             nn.ReLU(inplace=True))

    def forward(self, x, mode='train'):
        save_feature_map(x, "input")

        fpn_feat, deblur_img = self.deblur_FPN(x)
        save_feature_map(fpn_feat, "fpn_feat")
        save_feature_map(deblur_img, "fpn_feat")

        out = self.stage0(x)
        save_feature_map(out, "stage0")

        for blk in self.stage1:
            out = blk(out)
        save_feature_map(out, "stage1")
        for blk in self.stage2:
            out = blk(out)
        save_feature_map(out, "stage2")

        fpn_up = F.interpolate(fpn_feat, size=out.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.feature_fusion(out, fpn_up)
        save_feature_map(fused, "fused_before_AGE")
        fused = fused + self.AGE(fused)
        save_feature_map(fused, "fused_after_AGE")

        for blk in self.stage3:
            fused = blk(fused)
        save_feature_map(fused, "stage3")
        fused = self.deconv_layers1(fused)
        save_feature_map(fused, "deconv1")
        fused = self.deconv_layers2(fused)
        save_feature_map(fused, "deconv2")

        ret = {head: self.__getattr__(head)(fused) for head in self.heads}

        if mode == 'val':
            return [ret]
        elif mode == 'train':
            return [ret], deblur_img
        else:
            raise ValueError("mode not eq train/val!!!")


def create_DREB_FPN_Net_detect_SSAttn_v2(heads=None, head_conv=None):
    print('create DREB_Net_detect with improved FPN + S6 attention')
    return DREB_FPN_Net_SSAttn_v2(heads=heads, head_conv=head_conv)


##########################################
# ======== 测试 ========
##########################################
if __name__ == '__main__':
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    model = create_DREB_FPN_Net_detect_SSAttn_v2(heads=heads, head_conv=64).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    try:
        y, fpn_feat = model(x, mode='train')
        print('检测输出:', y[0]['hm'].shape, y[0]['wh'].shape, y[0]['reg'].shape)
        print('去模糊金字塔特征输出:', fpn_feat.shape)
    except Exception as e:
        print("前向执行发生异常:", e)
