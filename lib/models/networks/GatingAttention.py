# ======================================================
#  完整可运行：真实 SS2D（含 mamba_ssm）+ S6GatingAttention
# ======================================================


"""
选择性扫描门控注意力机制
实现机制：
1、先经过一个1X1的Conv实现投影操作
2、应用SS2D核心模块并且辅佐以MambaS6的核心扫描机制
3、最后经过一个Sigmoid函数并辅佐以1X1的Conv操作实现通道对其和参数权重归一化处理，最后与原始输入进行相乘，实现门控注意力机制的实现。
4、最重要的特征实现即插即用，且可以嵌入到任何网络模型中，输入和输出的特征保持不变，特征尺寸的大小保持不变。
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange, repeat

# 保留你的依赖
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


# ======================================================
# LayerNorm with Bias (你的原实现) 使用的参数初始化层次，
# ======================================================
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


# ======================================================
# SS2D（一字不改，使用你的实现） 服用Mamba注意力机制的核心机制，实现高效长短注意力
# ======================================================
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
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7, padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

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
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) *
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
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
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W
        K = 1

        x_hw = x.view(B, 1, -1, L)
        xs = x_hw

        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )

        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L), self.dt_projs_weight
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        return out_y[:, 0]

    def forward(self, x):
        # B,C,H,W → B,H,W,C
        x = rearrange(x, "b c h w -> b h w c")
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))

        y1 = self.forward_core(x)
        y = rearrange(y1, "b d (h w) -> b h w d", h=H, w=W)

        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)

        return rearrange(out, "b h w d -> b d h w")


# ======================================================
#   ★ S6GatingAttention (最终使用的注意力模块)
# ======================================================
class S6GatingAttention(nn.Module):
    """
    你的最终注意力版本：
    - pre_proj
    - SS2D 长短距离依赖增强
    - gate_conv → sigmoid
    - gated residual 输出
    """
    def __init__(self, channels):
        super().__init__()
        self.pre_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.s6 = SS2D(d_model=channels) # 利用SS6选择机制
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        # 应用门控卷积让模型更加聚集于陨石坑换装结构及其唤醒边界的特征提取，实现在局部安华，对比度下降的情况下的有效识别。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        proj = self.pre_proj(x)
        s6_out = self.s6(proj)
        gate = self.sigmoid(self.gate_conv(s6_out))
        return x * gate + x   # gated residual


# ======================================================
#   测试：224×224×3 示例
# ======================================================
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224).cuda()  # GPU 上运行
    attn = S6GatingAttention(channels=3).cuda()

    print("input:", x.shape)
    y = attn(x)
    print("output:", y.shape)
    print("✔ S6GatingAttention + SS2D forward success")
