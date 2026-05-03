import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels import inceptionresnetv2
import numpy as np


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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.se = SEBlock(out_channels, out_channels // 16)
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
# ======== 频域增强模块 LFAMM ========
##########################################
class LFAMM(nn.Module):
    def __init__(self, channels=128, height=32, weight=32):
        super(LFAMM, self).__init__()
        self.learnable_h = height
        self.learnable_w = np.floor(weight / 2).astype(int) + 1
        self.register_parameter('convolution', nn.Parameter(torch.rand(channels, self.learnable_h, self.learnable_w)))

    def forward(self, x):
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        x_amp_invariant = x_amp * self.convolution
        x_fft_invariant = x_amp_invariant * torch.exp(torch.tensor(1j) * x_pha)
        return torch.fft.irfftn(x_fft_invariant, dim=(-2, -1))


# 没有改边网络结构的详细设计
##########################################
# ======== FPN + InceptionResNetV2 ========
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
        enc0 = self.enc0(x)      # [B, 32, 128, 128]
        enc1 = self.enc1(enc0)   # [B, 64, 64, 64]
        enc2 = self.enc2(enc1)   # [B, 192, 32, 32]
        enc3 = self.enc3(enc2)   # [B, 1088, 16, 16]
        enc4 = self.enc4(enc3)   # [B, 2080, 8, 8]

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



class FPNInception(nn.Module):
    """金字塔去模糊分支：输出多尺度特征 + 重建图像"""
    def __init__(self, norm_layer, num_filters=128, num_filters_fpn=256):
        super().__init__()
        self.fpn = FPN(norm_layer, num_filters_fpn)

        self.smooth = nn.Sequential(
            nn.Conv2d(1024, num_filters, 3, padding=1),
            norm_layer(num_filters), nn.ReLU()
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            norm_layer(num_filters), nn.ReLU()
        )

        # 新增：生成去模糊重建图像的卷积层
        self.final = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # ---- FPN 提取 ----
        map0, map1, map2, map3, map4 = self.fpn(x)

        # ---- 统一上采样到 map1 的空间尺寸（避免 scale_factor 导致的累积误差） ----
        target_size = map1.shape[-2:]  # 以 map1 为中间对齐参考（可改为其它）
        map4 = F.interpolate(map4, size=target_size, mode='nearest')
        map3 = F.interpolate(map3, size=target_size, mode='nearest')
        map2 = F.interpolate(map2, size=target_size, mode='nearest')
        map1 = F.interpolate(map1, size=target_size, mode='nearest')

        # ---- 多尺度融合 ----
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))

        # ---- 显式对齐到 map0 的尺寸，再做和 map0 的融合，避免尺寸不一致 ----
        map0_size = map0.shape[-2:]
        smoothed = F.interpolate(smoothed, size=map0_size, mode='nearest')

        smoothed2 = self.smooth2(smoothed + map0)  # [B, C, H_map0, W_map0]

        # ---- 最后显式上采样到输入图 x 的尺寸（而非使用 scale_factor） ----
        up_feat = F.interpolate(smoothed2, size=x.shape[2:], mode='bilinear', align_corners=False)  # [B, C, H_in, W_in]
        final = self.final(up_feat)  # [B, 3, H_in, W_in]

        # ---- 保证与 x 同尺寸再相加（保险起见） ----
        if final.shape[2:] != x.shape[2:]:
            final = F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=False)

        deblur_img = torch.tanh(final) + x
        deblur_img = torch.clamp(deblur_img, -1, 1)

        # 一个是选择输入到目标检测的分支，一个是选择对于模糊图像进行处理的分支
        return smoothed2, deblur_img



##########################################
# ======== 主干 DREB_Net (改版) ========
##########################################
class DREB_FPN_Net(nn.Module):
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

        # 改：去模糊FPN金字塔分支（输出特征+图像）
        self.deblur_FPN = FPNInception(norm_layer=nn.BatchNorm2d, num_filters=128)

        # 融合 stage2 与 FPN 金字塔特征
        self.feature_fusion = MAGFF(channels=128)

        # 频域增强
        self.LFAMM = LFAMM(channels=128, height=32, weight=32)

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
        # ---- 去模糊金字塔特征 + 重建图像 ---- 一个是用于模糊图像恢复，一个是用于目标检测
        fpn_feat, deblur_img = self.deblur_FPN(x)  # [B,128,128,128], [B,3,256,256]

        # ---- 主干 ----
        out = self.stage0(x)
        for blk in self.stage1:
            out = blk(out)
        for blk in self.stage2:
            out = blk(out)  # [B,128,32,32]

        # ---- 特征融合 ----
        fpn_up = F.interpolate(fpn_feat, size=out.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.feature_fusion(out, fpn_up)

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
            return [ret],deblur_img  # 增加 deblur 图像返回
        else:
            raise ValueError("mode not eq train/val!!!")


def create_DREB_FPN_Net_detect(heads=None, head_conv=None):
    print('create crater DREB_Net_detect with FPN-based deblur-assisted fusion')
    return DREB_FPN_Net(heads=heads, head_conv=head_conv)


##########################################
# ======== 测试 ========
##########################################
if __name__ == '__main__':
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    model = create_DREB_FPN_Net_detect(heads=heads, head_conv=64)
    x = torch.randn(1, 3, 256, 256)
    y, fpn_feat = model(x, mode='train')
    print('检测输出:', y[0]['hm'].shape, y[0]['wh'].shape, y[0]['reg'].shape)
    print('去模糊金字塔特征输出:', fpn_feat.shape)
