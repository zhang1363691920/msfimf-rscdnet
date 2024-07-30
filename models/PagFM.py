import torch
import torch.nn as nn
import torch.nn.functional as F

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self. up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)



        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))


        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x

# 实例化 PagFM 模块
# in_channels = 128  # 输入特征图的通道数
# mid_channels = 64  # 中间特征图的通道数
# after_relu = False  # 是否在特征图上应用 ReLU 激活函数
# with_channel = False  # 是否进行通道维度的融合
# BatchNorm = nn.BatchNorm2d  # 使用的批归一化层
#
# pag_fm = PagFM(in_channels, mid_channels, after_relu, with_channel, BatchNorm)
#
# # 创建输入数据
# batch_size = 4
# height = 128
# width = 128
# x = torch.randn(batch_size, in_channels, height, width)
# y = torch.randn(batch_size, in_channels, height, width)
#
# # 前向传播
# output = pag_fm(x, y)
#
# # 输出结果的形状
# print(output.shape)
