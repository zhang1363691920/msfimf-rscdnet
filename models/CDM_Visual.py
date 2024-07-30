import torch
import torch.nn as nn
from models.ResNet import build_resnet as ResNet
from models.Ccat4 import CascadeModule
from models.PLLCBAM import ChannelAttention as CAM
from models.PLLCBAM import SpatialAttention as SAM
from models.BiToken import BiToken
from models.CrossAttention import CrossAttention
import os
import time
import cv2
import numpy as np
from torchvision import utils as vutils
# from models.Ccat2 import PAPPM as PAPPM
def draw_features(x, index, savename):
    savepath = 'visual/LEVIR/'+'/'+str(index)+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    tic = time.time()
    # 1.1 获取feature maps
    features = x  # 尺度大小，如：torch.Size([1,80,45,45])
    # 1.2 每个通道对应元素求和
    heatmap = torch.sum(features.cpu(), dim=1)  # 尺度大小， 如torch.Size([1,45,45])
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    heatmap = (heatmap - min_value) / (max_value - min_value) * 255
    heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)  # 尺寸大小，如：(45, 45, 1)
    src_size = (256, 256)
    heatmap = cv2.resize(heatmap, src_size, interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 保存热力图
    cv2.imwrite(savepath+savename, heatmap)
class CDM(nn.Module):
    '''
    网络整体结构
    '''
    def __init__(self, backbone='resnet34'):
        super(CDM, self).__init__()
        self.resnet = ResNet(backbone=backbone)
        self.softmax = nn.Softmax(dim=1)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.upsamplex16 = nn.Upsample(scale_factor=16, mode='bilinear')
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
        self.conv11 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # self.conv12 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )
        self.conv_result = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
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
        # self.pappm = PAPPM(inplanes=256, branch_planes=128, outplanes=256)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2)
        )
    def forward(self, input1, input2, index, label):     # torch.size([1, 3, 256, 256])
        # 编码
        # 1.特征提取
        input1_f1, input1_f2, input1_f3, input1_f4, input1_f5 = self.resnet(input1)
        input2_f1, input2_f2, input2_f3, input2_f4, input2_f5 = self.resnet(input2)

        draw_features(input1_f1, index, "input1_f1.png")
        draw_features(input1_f2, index, "input1_f2.png")
        draw_features(input1_f3, index, "input1_f3.png")
        draw_features(input1_f4, index, "input1_f4.png")
        draw_features(input1_f5, index, "input1_f5.png")

        draw_features(input2_f1, index, "input2_f1.png")
        draw_features(input2_f2, index, "input2_f2.png")
        draw_features(input2_f3, index, "input2_f3.png")
        draw_features(input2_f4, index, "input2_f4.png")
        draw_features(input2_f5, index, "input2_f5.png")

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

        draw_features(input1_fl1, index, "input1_fl1.png")
        draw_features(input1_fl2, index, "input1_fl2.png")
        draw_features(input1_fl3, index, "input1_fl3.png")
        draw_features(input1_fh4, index, "input1_fh4.png")
        draw_features(input1_fh5, index, "input1_fh5.png")

        draw_features(input2_fl1, index, "input2_fl1.png")
        draw_features(input2_fl2, index, "input2_fl2.png")
        draw_features(input2_fl3, index, "input2_fl3.png")
        draw_features(input2_fh4, index, "input2_fh4.png")
        draw_features(input2_fh5, index, "input2_fh5.png")
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
        draw_features(output1, index, "cat_1.png")
        draw_features(output2, index, "cat_2.png")
        draw_features(output3, index, "cat_3.png")
        draw_features(output4, index, "cat_4.png")
        draw_features(output5, index, "cat_5.png")
        x_list = [output1, output2, output3, output4, output5]
        inputs = list(zip(x_list, self.bit.conv_a))      # 将输入和卷积操作压缩成一个列表
        output_tokens = self.bit(inputs)         # 当做KV，输出是一个列表
        output1 = self.cra0(output1, output_tokens[0])  # torch.Size([1, 128, 128, 128])
        output2 = self.cra1(output2, output_tokens[1])  # torch.Size([1, 128, 64, 64])
        output3 = self.cra2(output3, output_tokens[2])  # torch.Size([1, 128, 32, 32])
        output4 = self.cra3(output4, output_tokens[3])  # torch.Size([1, 128, 16, 16])
        output5 = self.cra4(output5, output_tokens[4])  # torch.Size([1, 128, 8, 8])
        draw_features(output1, index, "cross_1.png")
        draw_features(output2, index, "cross_2.png")
        draw_features(output3, index, "cross_3.png")
        draw_features(output4, index, "cross_4.png")
        draw_features(output5, index, "cross_5.png")
        # 高级特征的全局信息获取
        output5_up2x = self.upsamplex2(output5)
        output5_up4x = self.upsamplex2(output5_up2x)
        output5_up8x = self.upsamplex2(output5_up4x)
        output5_up16x = self.upsamplex2(output5_up8x)
        out4 = torch.cat([output4, output5_up2x], dim=1)
        out3 = torch.cat([output3, output5_up4x], dim=1)
        out2 = torch.cat([output2, output5_up8x], dim=1)
        out1 = torch.cat([output1, output5_up16x], dim=1)
        draw_features(out1, index, "hfglf_1.png")
        draw_features(out2, index, "hfglf_2.png")
        draw_features(out3, index, "hfglf_3.png")
        draw_features(out4, index, "hfglf_4.png")
        # out5 = self.pappm(output5)
        #
        # output5_up = self.upsamplex2(output5)
        # out5_up = self.upsamplex2(out5)
        # out4 = torch.cat([output4, output5_up, out5_up], dim=1)
        # out4 = self.conv11(out4)
        #
        # out5_up = self.upsamplex2(out5_up)
        # out4_up = self.upsamplex2(out4)
        # out3 = torch.cat([output3, out4_up, out5_up], dim=1)
        # out3 = self.conv11(out3)
        #
        # out5_up = self.upsamplex2(out5_up)
        # out3_up = self.upsamplex2(out3)
        # out2 = torch.cat([output2, out3_up, out5_up], dim=1)
        # out2 = self.conv11(out2)
        #
        # out5_up = self.upsamplex2(out5_up)
        # out2_up = self.upsamplex2(out2)
        # out1 = torch.cat([output1, out2_up, out5_up], dim=1)
        # out1 = self.conv11(out1)
        # 多尺度融合
        result = self.cat(out4, out3, out2, out1)  # torch.Size([1, 128*5, 128, 128])
        # result = self.cat(output5, output4, output3, output2, output1)  # torch.Size([1, 128*5, 128, 128])
        result = self.conv_result(result)
        # 变化图生成
        result = self.upsamplex2(result)
        result = self.classifier(result)
        result = self.softmax(result)
        draw_features(result, index, "out.png")
        savepath = 'visual/LEVIR/' + str(index) + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        vutils.save_image(input1, 'visual/LEVIR/' + str(index) + '/input1.png')
        vutils.save_image(input2, 'visual/LEVIR/' + str(index) + '/input2.png')
        vutils.save_image(label, 'visual/LEVIR/' + str(index) + '/label.png')
        return result
# cdm = CDM()
# input1 = torch.randn(2, 3, 256, 256)
# input2 = torch.randn(2, 3, 256, 256)
# c1= cdm(input1, input2)
# print(c1.shape)