import torch
import torch.nn as nn


# 定义级联模块
class CascadeModule(nn.Module):
    def __init__(self):
        super(CascadeModule, self).__init__()
        # 初始化级联模块中的上采样操作
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, s1, s2, s3, s4, s5):
        # 对s1进行上采样
        s1_up = self.upsample(s1)

        # 将s1_up和s2级联
        s2_cat = s1_up + s2
        # s2_cat = torch.cat([s1_up, s2], dim=1)


        # 对s2_cat进行上采样
        s2_up = self.upsample(s2_cat)

        # 将s2_up和s3级联
        s3_cat = s2_up + s3
        # s3_cat = torch.cat([s2_up, s3], dim=1)

        # 对s3_cat进行上采样
        s3_up = self.upsample(s3_cat)

        # 将s3_up和s4级联
        s4_cat = s3_up + s4
        # s4_cat = torch.cat([s3_up, s4], dim=1)


        # 对s4_cat进行上采样
        s4_up = self.upsample(s4_cat)

        # 将s4_up和s5级联
        s5_cat = s4_up + s5
        # s5_cat = torch.cat([s4_up, s5], dim=1)


        return s5_cat

# # 实例化级联模块
# cdm = CascadeModule()
#
# # 假设您已经通过其他操作得到了s1-s5的值
#
# # 进行级联操作
# result = cdm.forward(s1, s2, s3, s4, s5)
#
# print(result.shape)
'''PIDnet
pag注意力引导同级特征图的融合
'''
# import torch.nn as nn
# import torch
# import torch.nn.functional as F
#
#
# class PagFM(nn.Module):
#     def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
#         super(PagFM, self).__init__()
#         self.with_channel = with_channel
#         self.after_relu = after_relu
#         self.f_x = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels,
#                       kernel_size=1, bias=False),
#             BatchNorm(mid_channels)
#         )
#         self.f_y = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels,
#                       kernel_size=1, bias=False),
#             BatchNorm(mid_channels)
#         )
#         if with_channel:
#             self.up = nn.Sequential(
#                 nn.Conv2d(mid_channels, in_channels,
#                           kernel_size=1, bias=False),
#                 BatchNorm(in_channels)
#             )
#         if after_relu:
#             self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, y):
#         input_size = x.size()
#         if self.after_relu:
#             y = self.relu(y)
#             x = self.relu(x)
#
#         y_q = self.f_y(y)
#         y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
#                             mode='bilinear', align_corners=False)
#         x_k = self.f_x(x)
#
#         if self.with_channel:
#             sim_map = torch.sigmoid(self.up(x_k * y_q))
#         else:
#             sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
#
#         y = F.interpolate(y, size=[input_size[2], input_size[3]],
#                           mode='bilinear', align_corners=False)
#         x = (1 - sim_map) * x + sim_map * y
#
#         return x
# 实例化 PagFM 模块
# in_channels = 64  # 输入特征图的通道数
# mid_channels = 32  # 中间特征图的通道数
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