import torch
import torch.nn as nn
from models.ResNet import build_resnet as ResNet
from models.Ccat import CascadeModule
from models.PLLCBAM import ChannelAttention as CAM
from models.PLLCBAM import SpatialAttention as SAM
from models.BiToken import BiToken
from models.CrossAttention import CrossAttention

class CDM(nn.Module):
    '''
    网络整体结构
    '''
    def __init__(self, backbone='resnet34'):
        super(CDM, self).__init__()
        self.resnet = ResNet(backbone=backbone)
        self.softmax = nn.Softmax(dim=1)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.cat = CascadeModule()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_result = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.sam = SAM()
        # self.cam1 = CAM(in_planes=128, ratio=16)
        self.cam2 = CAM(in_planes=256, ratio=16)
        self.cam3 = CAM(in_planes=512, ratio=16)
        self.bit = BiToken()
        self.cra0 = CrossAttention(input_dim=256)
        self.cra1 = CrossAttention(input_dim=256)
        self.cra2 = CrossAttention(input_dim=256)
        self.cra3 = CrossAttention(input_dim=256)
        self.cra4 = CrossAttention(input_dim=256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2)
        )
    def forward(self, input1, input2):     # torch.size([1, 3, 256, 256])
        # 编码
        # 1.特征提取
        input1_f1, input1_f2, input1_f3, input1_f4, input1_f5 = self.resnet(input1)
        input2_f1, input2_f2, input2_f3, input2_f4, input2_f5 = self.resnet(input2)

        # 2.深浅层特征增强
        input1_fl1 = self.sam(input1_f1) * input1_f1   # c = 64
        input1_fl2 = self.sam(input1_f2) * input1_f2   # c = 64
        input1_fl3 = self.sam(input1_f3) * input1_f3  # c = 128
        input1_fh4 = self.cam2(input1_f4) * input1_f4  # c = 256
        input1_fh5 = self.cam3(input1_f5) * input1_f5  # c = 512

        input2_fl1 = self.sam(input2_f1) * input2_f1
        input2_fl2 = self.sam(input2_f2) * input2_f2
        input2_fl3 = self.sam(input2_f3) * input2_f3
        input2_fh4 = self.cam2(input2_f4) * input2_f4
        input2_fh5 = self.cam3(input2_f5) * input2_f5
        # 高级语义特征
        output1 = torch.cat((input1_fl1, input2_fl1), dim=1)  # torch.size([1, 128, 128, 128])
        output2 = torch.cat((input1_fl2, input2_fl2), dim=1)  # torch.Size([1, 128, 64, 64])
        output3 = torch.cat((input1_fl3, input2_fl3), dim=1)  # torch.Size([1, 256, 32, 32])
        output4 = torch.cat((input1_fh4, input2_fh4), dim=1)  # torch.Size([1, 512, 16, 16])
        output5 = torch.cat((input1_fh5, input2_fh5), dim=1)  # torch.Size([1, 1024, 8, 8])

        output1 = self.conv5(output1)  # torch.size([1, 256, 128, 128])
        output2 = self.conv4(output2)  # torch.size([1, 256, 64, 64])
        output3 = self.conv3(output3)  # torch.size([1, 256, 32, 32])
        output4 = self.conv2(output4)  # torch.size([1, 256, 16, 16])
        output5 = self.conv1(output5)  # torch.size([1, 256, 8, 8])

        x_list = [output1, output2, output3, output4, output5]
        inputs = list(zip(x_list, self.bit.conv_a))      # 将输入和卷积操作压缩成一个列表
        output_tokens = self.bit.forward(inputs)         # 当做KV，输出是一个列表
        output1 = self.cra0.forward(output1, output_tokens[0])  # torch.Size([1, 128, 128, 128])
        output2 = self.cra1.forward(output2, output_tokens[1])  # torch.Size([1, 128, 64, 64])
        output3 = self.cra2.forward(output3, output_tokens[2])  # torch.Size([1, 128, 32, 32])
        output4 = self.cra3.forward(output4, output_tokens[3])  # torch.Size([1, 128, 16, 16])
        output5 = self.cra4.forward(output5, output_tokens[4])  # torch.Size([1, 128, 8, 8])

        # 残差模块
        # output_cra1 = self.conv0(torch.cat([output_cra1, output1], dim=1))  # torch.Size([1, 96, 128, 128])
        # output_cra2 = self.conv1(torch.cat([output_cra2, output2], dim=1))  # torch.Size([1, 96, 64, 64])
        # output_cra3 = self.conv2(torch.cat([output_cra3, output3], dim=1))  # torch.Size([1, 192, 32, 32])
        # output_cra4 = self.conv3(torch.cat([output_cra4, output4], dim=1))  # torch.Size([1, 384, 16, 16])
        # output_cra5 = self.conv4(torch.cat([output_cra5, output5], dim=1))  # torch.Size([1, 768, 8, 8])

        # 多尺度融合
        result = self.cat.forward(output5, output4, output3, output2, output1)  # torch.Size([1, 128*5, 128, 128])
        result = self.conv_result(result)
        # 变化图生成
        result = self.upsamplex2(result)
        result = self.classifier(result)
        result = self.softmax(result)
        return result
# cdm = CDM()
# input1 = torch.randn(1, 3, 256, 256)
# input2 = torch.randn(1, 3, 256, 256)
# c1= cdm.forward(input1, input2)