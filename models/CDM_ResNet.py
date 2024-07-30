import torch
import torch.nn as nn
from models.ResNet import build_resnet as ResNet
from models.Ccat import CascadeModule
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

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2)
        )
    def forward(self, input1, input2):                      # torch.size([1, 3, 256, 256])
        # 编码
        # 1.特征提取
        input1_f1, input1_f2, input1_f3, input1_f4, input1_f5 = self.resnet(input1)
        input2_f1, input2_f2, input2_f3, input2_f4, input2_f5 = self.resnet(input2)

        output1 = torch.cat((input1_f1, input2_f1), dim=1)  # torch.size([1, 128, 128, 128])
        output2 = torch.cat((input1_f2, input2_f2), dim=1)  # torch.Size([1, 128, 64, 64])
        output3 = torch.cat((input1_f3, input2_f3), dim=1)  # torch.Size([1, 256, 32, 32])
        output4 = torch.cat((input1_f4, input2_f4), dim=1)  # torch.Size([1, 512, 16, 16])
        output5 = torch.cat((input1_f5, input2_f5), dim=1)  # torch.Size([1, 1024, 8, 8])

        output1 = self.conv5(output1)  # torch.size([1, 128, 128, 128])
        output2 = self.conv4(output2)  # torch.size([1, 128, 64, 64])
        output3 = self.conv3(output3)  # torch.size([1, 128, 32, 32])
        output4 = self.conv2(output4)  # torch.size([1, 128, 16, 16])
        output5 = self.conv1(output5)  # torch.size([1, 128, 8, 8])

        # 多尺度融合
        result = self.cat.forward(output5, output4, output3, output2, output1)  # torch.Size([1, 128*5, 128, 128])
        result = self.conv_result(result)
        # 变化图生成
        result = self.upsamplex2(result)
        result = self.classifier(result)
        result = self.softmax(result)
        return result

# cdm = CDM()
# input1 = torch.randn(2, 3, 256, 256)
# input2 = torch.randn(2, 3, 256, 256)
# c1= cdm(input1, input2)
# print(c1.shape)















