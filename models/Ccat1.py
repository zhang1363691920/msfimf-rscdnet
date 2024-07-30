import torch
import torch.nn as nn
from models.PagFM import PagFM as PagFM
# 定义级联模块
class CascadeModule(nn.Module):
    def __init__(self):
        super(CascadeModule, self).__init__()
        # 初始化级联模块中的上采样操作
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pagfm = PagFM(in_channels=256, mid_channels=128)
    def forward(self, s1, s2, s3, s4, s5):
        s2 = self.pagfm(s2, s1)
        s3 = self.pagfm(s3, s1)
        s4 = self.pagfm(s4, s1)
        s5 = self.pagfm(s5, s1)

        s2_up = self.upsample(s2)
        # s3_cat = torch.cat([s2_up, s3], dim=1)
        s3_cat = s2_up + s3

        s3_up = self.upsample(s3_cat)
        # s4_cat = torch.cat([s3_up, s4], dim=1)
        s4_cat = s3_up + s4

        s4_up = self.upsample(s4_cat)
        # s5_cat = torch.cat([s4_up, s5], dim=1)
        s5_cat = s4_up + s5
        return s5_cat
    # def forward(self, s1, s2, s3, s4, s5):
    #
    #     # 对s1进行上采样
    #     s1_up = self.upsample(s1)
    #     # 将s1_up和s2级联
    #     s2_cat = torch.cat([s1_up, s2], dim=1)
    #     # s2_cat = s1_up + s2
    #
    #     # 对s2_cat进行上采样
    #     s2_up = self.upsample(s2_cat)
    #     # 将s2_up和s3级联
    #     s3_cat = torch.cat([s2_up, s3], dim=1)
    #     # s3_cat = s2_up + s3
    #
    #     # 对s3_cat进行上采样
    #     s3_up = self.upsample(s3_cat)
    #     # 将s3_up和s4级联
    #     s4_cat = torch.cat([s3_up, s4], dim=1)
    #     # s4_cat = s3_up + s4
    #
    #
    #     # 对s4_cat进行上采样
    #     s4_up = self.upsample(s4_cat)
    #     # 将s4_up和s5级联
    #     s5_cat = torch.cat([s4_up, s5], dim=1)
    #     # s5_cat = s4_up + s5
    #
    #     return s5_cat

# # 实例化级联模块
# cdm = CascadeModule()
#
# # 假设您已经通过其他操作得到了s1-s5的值
#
# # 进行级联操作
# result = cdm.forward(s1, s2, s3, s4, s5)
#
# print(result.shape)
