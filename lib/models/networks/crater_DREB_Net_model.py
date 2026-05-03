import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock_useSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, deploy=None,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(RepVGGBlock_useSE, self).__init__()
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)

        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class Deblur_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Deblur_Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            # 对齐尺寸（以x2为基准）
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class MAGFF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(MAGFF, self).__init__()
        internal_channels = int(channels // r)

        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1, 1, 0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )

        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, internal_channels, 1, 1, 0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )

        self.local_attention_2 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1, 1, 0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )
        
        self.global_attention_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, internal_channels, 1, 1, 0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x_a = x1 + x2
        x_l = self.local_attention(x_a)
        x_g = self.global_attention(x_a)
        x_lg = x_l + x_g
        weight = self.sigmoid(x_lg)
        x_i = x1 * weight + x2 * (1 - weight)

        x_l2 = self.local_attention_2(x_i)
        x_g2 = self.global_attention_2(x_i)
        x_lg2 = x_l2 + x_g2
        weight2 = self.sigmoid(x_lg2)
        x_output = x1 * weight2 + x2 * (1 - weight2)
        return x_output


class LFAMM(nn.Module):
    def __init__(self, channels=128, height=32, weight=32):  # 修改为适配256输入
        super(LFAMM, self).__init__()
        self.channels = channels
        self.height = height
        self.weight = weight
        self.learnable_h = self.height
        self.learnable_w = np.floor(self.weight/2).astype(int) + 1
        self.register_parameter('convolution', torch.nn.Parameter(torch.rand(self.channels, self.learnable_h, self.learnable_w), requires_grad=True))

    def forward(self, x):
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))
        x_fft = x_fft + 1e-8
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        x_amp_invariant = torch.mul(x_amp, self.convolution)
        x_fft_invariant = x_amp_invariant * torch.exp(torch.tensor(1j) * x_pha)
        x_invariant = torch.fft.irfftn(x_fft_invariant, dim=(-2, -1))
        return x_invariant


class DREB_Net(nn.Module):
    def __init__(self, num_blocks=[3, 5, 8], width_multiplier=[1, 1, 1], override_groups_map=None, deploy=False, use_checkpoint=False,
                 heads=None, head_conv=None):
        super(DREB_Net, self).__init__()
        self.deconv_with_bias = False
        self.heads = heads
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_checkpoint = use_checkpoint

        # 调整LFAMM输入维度
        self.LFAMM = LFAMM(channels=128, height=32, weight=32)

        # Backbone 部分（减少stage数量）
        self.in_planes = 64
        self.stage0 = RepVGGBlock_useSE(3, 64, 3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(64, num_blocks[0], stride=2)
        self.stage2 = self._make_stage(128, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(256, num_blocks[2], stride=2)

        # 减少 deconv 数量
        self.deconv_layers1 = self._make_deconv_layer(256, 4)
        self.deconv_layers2 = self._make_deconv_layer(256, 4)

        # Deblur 分支同步简化
        self.deblur_down1 = Deblur_Down(64, 64)
        self.deblur_down2 = Deblur_Down(64, 128)
        self.deblur_down3 = Deblur_Down(128, 256)
        self.deblur_up1 = Deblur_Up(256, 256, 128)
        self.deblur_up2 = Deblur_Up(128, 128, 64)
        self.deblur_up3 = Deblur_Up(64, 96, 3)

        self.MAGFF_attention = MAGFF(channels=128)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(in_channels=256, out_channels=num_output, kernel_size=1, stride=1, padding=0)
            self.__setattr__(head, fc)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock_useSE(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                            stride=stride, padding=1, deploy=self.deploy))
            self.in_planes = planes
        return nn.ModuleList(blocks)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding, output_padding = 1, 0
        elif deconv_kernel == 3:
            padding, output_padding = 1, 1
        else:
            padding, output_padding = 0, 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, deconv_kernel):
        kernel, padding, output_padding = self._get_deconv_cfg(deconv_kernel)
        planes = num_filters
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.in_planes, out_channels=planes,
                               kernel_size=kernel, stride=2, padding=padding,
                               output_padding=output_padding, bias=self.deconv_with_bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.in_planes = planes
        return layer

    def forward(self, x, mode):
        out = self.stage0(x)
        s0 = out
        s1 = self.deblur_down1(s0)
        s2 = self.deblur_down2(s1)

        for block in self.stage1:
            out = block(out)
        for block in self.stage2:
            out = block(out)

        out_LFAMM = self.LFAMM(out)
        out = self.MAGFF_attention(out, s2)
        out = out + out_LFAMM

        for block in self.stage3:
            out = block(out)

        out = self.deconv_layers1(out)
        out = self.deconv_layers2(out)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(out)

        if mode == 'val':
            return [ret]
        elif mode == 'train':
            deblur_inp = s2
            down3 = self.deblur_down3(deblur_inp)
            up1 = self.deblur_up1(down3, s2)
            up2 = self.deblur_up2(up1, s1)
            deblur_out = self.deblur_up3(up2, s0)

            if deblur_out.shape[2:] != x.shape[2:]:
                deblur_out = F.interpolate(deblur_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return [ret], deblur_out
        else:
            raise ValueError("mode not eq train/val!!!")


def create_DREB_Net_detect(deploy=False, use_checkpoint=False, heads=None, head_conv=None):
    print('create_DREB_Net_detect (for 256×256 input)')
    return DREB_Net(override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,
                    heads=heads, head_conv=head_conv)


if __name__ == '__main__':
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    head_conv = 64
    model = create_DREB_Net_detect(heads=heads, head_conv=head_conv)
    input_size = 256
    x = torch.randn(2, 3, input_size, input_size)
    print(x.shape)

    mode = 'val'
    if mode == 'train':
        y, deblur_out = model(x, mode)
        print(y[0]['hm'].shape, y[0]['wh'].shape, y[0]['reg'].shape, deblur_out.shape)
    else:
        y = model(x, mode)
        print(y[0]['hm'].shape, y[0]['wh'].shape, y[0]['reg'].shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params/1e6:.2f} M')

    from thop import profile
    macs, params = profile(model, inputs=(torch.randn(1, 3, input_size, input_size), 'val'), verbose=False)
    print(f'FLOPs: {macs/1e9:.2f} G')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    import time
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor, mode='val')

    t0 = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(input_tensor, mode='val')
    t1 = time.time()
    print(f'平均推理时间: {(t1-t0)/50*1000:.2f} ms')
