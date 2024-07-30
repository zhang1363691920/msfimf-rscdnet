#CAM通道注意力
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# if __name__ == '__main__':
#     CA = ChannelAttention(32)
#     data_in = torch.randn(8, 32, 300, 300)
#     data_out = CA(data_in)
#     print(data_in.shape)  # torch.Size([8, 32, 300, 300])
#     print(data_out.shape)  # torch.Size([8, 32, 1, 1])

#SAM空间注意力
# import torch
# from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        #默认kernel_size是7时，padding是3，否则为1
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #torch.max返回的是值和索引，所以max_out,_这种形式接受值，忽略索引
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# if __name__ == '__main__':
#     SA = SpatialAttention(7)
#     data_in = torch.randn(8, 32, 300, 300)
#     data_out = SA(data_in)
#     print(data_in.shape)  # torch.Size([8, 32, 300, 300])
#     print(data_out.shape)  # torch.Size([8, 1, 300, 300])

#CBAM
# import torch
# import torch.nn as nn
#
#
# class CBAMLayer(nn.Module):
#     def __init__(self, channel, reduction=16, spatial_kernel=7):
#         super(CBAMLayer, self).__init__()
#
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
#                               padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x
#
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x
#
#
# x = torch.randn(1, 1024, 32, 32)
# net = CBAMLayer(1024)
# y = net.forward(x)
# print(y.shape)