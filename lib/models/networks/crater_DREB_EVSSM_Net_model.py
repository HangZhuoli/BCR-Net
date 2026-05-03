# -*- coding: utf-8 -*-
"""
1、使用EVSSM代替原有的金字塔，实现对于图像的去模糊操作
2、使用MAGFF模块，实现特征融合操作，将模糊图像的特征和清晰图像的特征进行融合，实现特征交换和互补操作
3、使用SEBlock模块，实现特征增强操作，对特征进行增强，提高模型的性能，主要在于强化主干网络中目标检测分支的识别性能
4、使用RepVGGBlock模块内部嵌入SEBlock模块实现特征提取操作

核心：利用Mamba编码长短距离依赖以及非空间依赖 ----》 实现利用其为一种注意力机制进行特征增强然后应用FPN网络中

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
import numpy as np

# 下面这个 import 依赖于 mamba_ssm 库，你的环境需要安装该包（你在原始片段中已引用它）
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

# pretrained inceptionresnetv2
from pretrainedmodels import inceptionresnetv2

##########################################
# ======== 基础模块 ========
##########################################
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
    # 使用RepVGGBlock的结构，但是增加了SEBlock模块，用于增强特征，用于主干网络的特征提取实现对于目标检测分支的前期操作
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
        inter = max(channels // r, 1)
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
# ======== 频域增强模块 LFAMM ========
##########################################
class LFAMM(nn.Module):
    """
    修改版：自动根据输入通道匹配，并确保参数在正确的设备上
    """
    def __init__(self, height=32, width=32):
        super(LFAMM, self).__init__()
        self.learnable_h = height
        self.learnable_w = int(np.floor(width / 2)) + 1

        # 占位，第一次 forward 时根据输入自动初始化
        self.register_parameter('convolution', None)

    def _init_conv(self, C, device):
        """
        根据输入通道数 C 动态初始化可学习 FFT mask，并放到正确 device
        """
        conv = nn.Parameter(
            torch.rand(C, self.learnable_h, self.learnable_w, device=device)
        )
        self.convolution = conv

    def forward(self, x):
        B, C, H, W = x.shape

        # -------- 首次执行时初始化权重到正确的设备 --------
        if self.convolution is None:
            self._init_conv(C, x.device)

        # -------- FFT --------
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)

        # -------- 对齐卷积核尺寸（中心裁剪）--------
        conv = self.convolution
        conv_h, conv_w = conv.shape[1:]

        conv = conv[:, :min(conv_h, x_amp.size(2)), :min(conv_w, x_amp.size(3))]

        # FFT amplitude mask
        x_amp[..., :conv.shape[1], :conv.shape[2]] *= conv.unsqueeze(0)

        # -------- 复数重建 --------
        x_fft_new = x_amp * torch.exp(
            torch.complex(torch.zeros_like(x_pha), x_pha)
        )
        out = torch.fft.irfftn(x_fft_new, dim=(-2, -1))

        return out




##########################################
# ======== FPN (保留用于 backbone 特征) ======== 原始FPN网络设计，与当下没有关系。
##########################################
class FPN(nn.Module):
    def __init__(self, norm_layer, num_filters=256):
        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(self.inception.conv2d_2a,
                                  self.inception.conv2d_2b,
                                  self.inception.maxpool_3a)
        self.enc2 = nn.Sequential(self.inception.conv2d_3b,
                                  self.inception.conv2d_4a,
                                  self.inception.maxpool_5a)
        self.enc3 = nn.Sequential(self.inception.mixed_5b,
                                  self.inception.repeat,
                                  self.inception.mixed_6a)
        self.enc4 = nn.Sequential(self.inception.repeat_1,
                                  self.inception.mixed_7a)

        self.lateral4 = nn.Conv2d(2080, num_filters, 1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, 1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, 1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, 1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, 1, bias=False)

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                 norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                 norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                 norm_layer(num_filters), nn.ReLU(inplace=True))

    def forward(self, x):
        enc0 = self.enc0(x)      # [B, 32, ...]
        enc1 = self.enc1(enc0)   # [B, 64, ...]
        enc2 = self.enc2(enc1)   # [B, 192, ...]
        enc3 = self.enc3(enc2)   # [B, 1088, ...]
        enc4 = self.enc4(enc3)   # [B, 2080, ...]

        lat4 = self.lateral4(enc4)
        lat3 = self.lateral3(enc3)
        lat2 = self.lateral2(enc2)
        lat1 = self.lateral1(enc1)
        lat0 = self.lateral0(enc0)

        map4 = lat4
        map3 = self.td1(lat3 + F.interpolate(map4, size=lat3.shape[-2:], mode='nearest'))
        map2 = self.td2(lat2 + F.interpolate(map3, size=lat2.shape[-2:], mode='nearest'))
        map1 = self.td3(lat1 + F.interpolate(map2, size=lat1.shape[-2:], mode='nearest'))

        return lat0, map1, map2, map3, map4


##########################################
# ======== EVSSM 及其组件（基于你提供的实现） ======== 实现核心的将二维的图像转换为一维的序列进行空间关系的捕获
##########################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        return x

# Mamba选择性扫描机制的核心算法
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        # pack x_proj weights for efficiency (K=1 here)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
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

    def forward_core(self, x: torch.Tensor):
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

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')

        return out

class EVS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', att=False, idx=3, patch=128):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)
        self.kernel_size = (patch, patch)

    def forward(self, x):
        if self.att:
            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

class EVSSM(nn.Module):
    """
    精简版 EVSSM:
        - 仅 2 层 encoder
        - 仅 1 层 decoder
        - decoder_level1 作为输出
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 2],     # 前两层编码
                 ffn_expansion_factor=3,
                 bias=False,
                 ) -> None:
        super(EVSSM, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ---------- Encoder Level1 ----------
        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, att=True, idx=i, patch=384)
            self.encoder_level1.add_module(f"block{i}", block)

        # ---------- Encoder Level2 ----------
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, att=True, idx=i, patch=192)
            self.encoder_level2.add_module(f"block{i}", block)

        # ---------- Decoder （只保留 Level2 → Level1） ----------
        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):  # 解码层与 Encoder1 对应深度一致
            block = EVS(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, att=True, idx=i, patch=384)
            self.decoder_level1.add_module(f"block{i}", block)

        # ---------- 输出层 ----------
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        # ===== Encoder Level 1 =====
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # ===== Encoder Level 2 =====
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # ===== Decoder 仅 1 层 =====
        inp_dec_level1 = self.up2_1(out_enc_level2)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1  # skip-connection

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        decoder_feature = out_dec_level1

        # ===== 输出去模糊图 =====
        deblur_img = self.output(out_dec_level1)
        deblur_img = torch.clamp(deblur_img + inp_img, -1, 1)

        return decoder_feature, deblur_img



##########################################
# ======== 将 EVSSM 用作金字塔 deblur 分支的封装 ========
##########################################
class DeblurEVSSMWrapper(nn.Module):
    """
    作为 FPNInception 的替代，返回 (decoder_feature, deblur_img)
    decoder_feature 的通道数应与 MAGFF 的 channels 参数一致（这里我们在创建时会设置 dim=128）
    """
    def __init__(self, dim=64, out_channels=3, num_blocks=[2,2,4], ffn_expansion_factor=3):
        super(DeblurEVSSMWrapper, self).__init__()
        self.evs_sm = EVSSM(inp_channels=3, out_channels=out_channels, dim=dim,
                            num_blocks=num_blocks, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        decoder_feature, deblur_img = self.evs_sm(x)
        return decoder_feature, deblur_img  # 与金字塔的实现方式不一样


##########################################
# ======== 主干 DREB_FPN_Net (改版) ========
##########################################
class DREB_EVSSM_Net(nn.Module):
    def __init__(self, heads=None, head_conv=64):
        super().__init__()
        self.heads = heads
        self.in_planes = 64

        # backbone lightweight stem / stages (RepVGG with SE)
        self.stage0 = RepVGGBlock_useSE(3, 64, 3, stride=2)
        self.stage1 = self._make_stage(64, 3, stride=2)
        self.stage2 = self._make_stage(64, 5, stride=2)
        self.stage3 = self._make_stage(256, 8, stride=2)

        self.deconv_layers1 = self._make_deconv_layer(256, 4)
        self.deconv_layers2 = self._make_deconv_layer(256, 4)

        # 改：用 EVSSM 作为去模糊分支（decoder_level1 的输出将用于融合）
        # 确保 dim 与 MAGFF(channels=128) 一致
        self.deblur_EVSSM = DeblurEVSSMWrapper(dim=64, out_channels=3, num_blocks=[2,2,4])

        # 融合 stage2(128ch) 与 EVSSM 的 decoder_feature (也为128ch)
        self.feature_fusion = MAGFF(channels=64)

        # 频域增强
        self.LFAMM = LFAMM(height=32, width=32)

        # heads
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
        return nn.Sequential(
            nn.ConvTranspose2d(self.in_planes, planes, kernel_size=kernel, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mode='train'):
        # ---- 去模糊分支（EVSSM）: decoder_feature (B,128,H_in,W_in), deblur_img (B,3,H_in,W_in)
        decoder_feat, deblur_img = self.deblur_EVSSM(x)

        # ---- 主干 ----
        out = self.stage0(x)
        for blk in self.stage1:
            out = blk(out)
        for blk in self.stage2:
            out = blk(out)  # [B,128, H_small, W_small] 例如 [B,128,32,32]

        # ---- 特征融合: 将 decoder_feat 上采样/下采样到与 out 相同的空间尺寸，然后用 MAGFF 融合 ----
        fpn_up = F.interpolate(decoder_feat, size=out.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.feature_fusion(out, fpn_up)  # 保持通道数为128

        # ---- 频域增强 ----
        fused = fused + self.LFAMM(fused)

        # ---- 检测头 ----
        for blk in self.stage3:
            fused = blk(fused)
        fused = self.deconv_layers1(fused)
        fused = self.deconv_layers2(fused)

        ret = {head: self.__getattr__(head)(fused) for head in self.heads}

        if mode == 'val':
            return [ret]
        elif mode == 'train':
            # 返回检测输出与去模糊图像（保持接口与原来一致）
            return [ret], deblur_img
        else:
            raise ValueError("mode not eq train/val!!!")


def create_DREB_EVSSM_Net_detect(heads=None, head_conv=None):
    print('create crater DREB_Net_detect with EVSSM-based deblur-assisted fusion')
    return DREB_EVSSM_Net(heads=heads, head_conv=head_conv)


##########################################
# ======== 测试 ========
##########################################
if __name__ == '__main__':
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    model = create_DREB_EVSSM_Net_detect(heads=heads, head_conv=64)
    x = torch.randn(1, 3, 256, 256)
    # 可能因为 mamba_ssm/selective_scan_fn 在当前环境未安装或初始化不同导致实际执行报错，
    # 但模型前向签名和返回值已按你的需求进行了调整
    try:
        y, fpn_feat = model(x, mode='train')
        print('检测输出 shapes:', y[0]['hm'].shape, y[0]['wh'].shape, y[0]['reg'].shape)
        print('去模糊金字塔特征输出:', fpn_feat.shape)
    except Exception as e:
        print("前向执行发生异常（环境依赖或初始化问题）:", e)
        # 仍然打印模型结构的关键信息
        print("模型创建成功。deblur_FPN 类型:", type(model.deblur_EVSSM))
        # 说明 expected shapes
        print("期望 decoder_feature channels = 128 (用于与检测分支融合)，deblur_img channels = 3.")

