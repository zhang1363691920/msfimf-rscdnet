import torch.nn as nn

"""
This code refers to "Pyramid attention network for semantic segmentation", that is
"https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch/blob/f719365c1780f062058dd0c94550c6c4766cd937/networks.py#L41"
"""

class FPM(nn.Module):
    def __init__(self, channels=1024):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPM, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Feature Pyramid
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """

        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)          # wenti
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        out = self.relu(x_master)

        return out
# 定义级联模块
class CascadeModule(nn.Module):
    def __init__(self):
        super(CascadeModule, self).__init__()
        # 初始化级联模块中的上采样操作
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, s1, s2, s3, s4, s5):
        # 对s1进行上采样
        # s1_up = self.upsample(s1)
        s1_up = s1
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