import torch.nn as nn

class CascadeModule(nn.Module):
    def __init__(self):
        super(CascadeModule, self).__init__()
        # 初始化级联模块中的上采样操作
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, s1, s2, s3, s4):


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
        # s4_up = self.upsample(s4_cat)
        #
        # # 将s4_up和s5级联
        # s5_cat = s4_up + s5
        # s5_cat = torch.cat([s4_up, s5], dim=1)


        return s4_cat