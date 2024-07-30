import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch import Tensor

class DWConv3x3BNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(DWConv3x3BNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )

class GhostModule(nn.Module):
    def __init__(self, in_channel, out_channel, s=2, kernel_size=1, stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channel = out_channel // s
        ghost_channel = intrinsic_channel * (s - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intrinsic_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(intrinsic_channel),
            nn.ReLU(inplace=True) if use_relu else nn.Sequential()
        )

        self.cheap_op = DWConv3x3BNReLU(in_channel=intrinsic_channel, out_channel=ghost_channel, stride=stride, groups=intrinsic_channel)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class HRViTFusionBlock(nn.Module):
    def __init__(
        self,
        # in_channels: Tuple[int] = (64, 64, 128, 256, 512),
        # out_channels: Tuple[int] = (64, 64, 128, 256, 512),
        in_channels: Tuple[int] = (128, 128, 128, 128, 128),
        out_channels: Tuple[int] = (128, 128, 128, 128, 128),
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ) -> None:
        # 初始化函数，设置输入输出通道数，激活函数，以及是否使用cp
        super().__init__()  # 继承父类nn.Module的属性
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.act_func = act_func  # 激活函数，默认为GELU
        self.with_cp = with_cp  # 是否使用cp，默认为False
        self.n_outputs = len(out_channels)  # 输出的数量
        self._build_fuse_layers()  # 调用函数，构建融合层

    def _build_fuse_layers(self):
        # 构建融合层，这些层用于将不同分辨率的特征融合在一起
        self.blocks = nn.ModuleList([])  # 初始化一个模块列表，用于存储所有的融合层
        self.ghostmodule = GhostModule(in_channel=128, out_channel=128)
        n_inputs = len(self.in_channels)  # 输入通道的数量
        for i, outc in enumerate(self.out_channels):  # 遍历每个输出通道
            blocks = nn.ModuleList([])  # 初始化一个模块列表，用于存储当前输出通道的所有融合层

            start = 0  # 开始的索引
            end = n_inputs  # 结束的索引
            for j in range(start, end):  # 遍历每个输入通道
                inc = self.in_channels[j]  # 当前输入通道的通道数
                if j == i:
                    # 对于相同分辨率的输入，不做任何处理，直接添加一个恒等映射
                    # blocks.append(nn.Identity())
                    blocks.append(self.ghostmodule)
                elif j < i:
                    # 对于分辨率较高的输入，进行下采样并增加通道数

                    block = [
                        # Depthwise Convolution，进行下采样
                        nn.Conv2d(
                            inc,
                            inc,
                            kernel_size=2 ** (i - j) + 1,
                            stride=2 ** (i - j),
                            dilation=1,
                            padding=2 ** (i - j) // 2,
                            groups=inc,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inc),  # 批量归一化
                        # Pointwise Convolution，增加通道数
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),  # 批量归一化
                    ]

                    blocks.append(nn.Sequential(*block))  # 将当前的融合层添加到列表中

                else:
                    # 对于分辨率较低的输入，减少通道数并进行上采样

                    block = [
                        # Pointwise Convolution，减少通道数
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),  # 批量归一化
                    ]

                    # 上采样
                    block.append(
                        nn.Upsample(
                            scale_factor=2 ** (j - i),
                            mode="nearest",
                        ),
                    )
                    blocks.append(nn.Sequential(*block))  # 将当前的融合层添加到列表中
            self.blocks.append(blocks)  # 将当前输出通道的所有融合层添加到列表中

        self.act = nn.ModuleList([self.act_func() for _ in self.out_channels])  # 对每个输出通道，都添加一个激活函数

    def forward(
        self,
        x: Tuple[
            Tensor,
        ],
    ) -> Tuple[Tensor,]:

        # 前向传播函数，对输入的每个特征图应用相应的融合层，然后将结果相加
        out = [None] * len(self.blocks)  # 初始化输出的列表（这里的out是一个self.block长度相同的空列表）
        n_inputs = len(x)  # 输入的数量

        for i, (blocks, act) in enumerate(zip(self.blocks, self.act)):  # 遍历每个输出通道的融合层和激活函数
            start = 0  # 开始的索引
            end = n_inputs  # 结束的索引
            for j, block in zip(range(start, end), blocks):  # 遍历输入通道的索引和对应的融合层，遍历每个输入通道的融合层
                out[i] = block(x[j]) if out[i] is None else out[i] + block(x[j])  # 应用融合层，并将结果相加
            out[i] = act(out[i])  # 应用激活函数
            # 更新输入数据，将当前层的输出替换到输入数据中
            x = (*x[:i], out[i], *x[i + 1:])
        return out  # 返回输出，输出是一个列表
# fusion_block = HRViTFusionBlock(
#     in_channels=(128, 128, 128, 128, 128),
#     out_channels=(128, 128, 128, 128, 128),
#     act_func=nn.ReLU,
# )
# features = (
#     torch.randn(1, 128, 64, 64),
#     torch.randn(1, 128, 32, 32),
#     torch.randn(1, 128, 16, 16),
#     torch.randn(1, 128, 8, 8),
#     torch.randn(1, 128, 4, 4),
# )
# out = fusion_block(features)
# print(out[0].shape)
# print(out[1].shape)
# print(out[2].shape)
# print(out[3].shape)
# print(out[4].shape)