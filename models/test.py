# import torch
# import torch.nn as nn
#
# class PAGModule(nn.Module):
#     def __init__(self, num_inputs):
#         super(PAGModule, self).__init__()
#         self.num_inputs = num_inputs
#         self.attention = nn.Sequential(
#             nn.Conv2d(num_inputs, 1, kernel_size=1),  # 用于生成注意力图的卷积层
#             nn.Sigmoid()  # 注意力图的激活函数
#         )
#
#     def forward(self, inputs):
#         attention_map = self.attention(inputs)  # 生成注意力图
#         weighted_inputs = inputs * attention_map  # 将输入特征图与注意力图相乘
#         fused_output = torch.sum(weighted_inputs, dim=1)  # 在通道维度上求和得到融合输出
#         return fused_output,attention_map,weighted_inputs
#
# # 示例用法
# num_inputs = 3  # 输入特征图的数量
# input_size = (5, 5)  # 输入特征图的尺寸
# batch_size = 16  # 批量大小
#
# # 创建PAG模块实例
# pag_module = PAGModule(num_inputs)
#
# # 生成随机输入特征图
# inputs = torch.randn(batch_size, num_inputs, input_size[0], input_size[1])
#
# # 前向传播
# fused_output,attention_map,weighted_inputs = pag_module(inputs)
# print("原始特征图",inputs.shape)#16,3,5,5
# print("卷积后的注意力图",attention_map.shape)#16,1,5,5
# print("注意力图乘原始特征图",weighted_inputs.shape)#16,3,5,5
# print("通道相加融合后",fused_output.shape)  # 打印融合输出的形状16,5,5
# print(fused_output)#打印特征图
#
# import torch
# import torch.nn as nn
#
#
# class CrossAttention1(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossAttention1, self).__init__()
#         self.query = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.key = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.value = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#         x2 = x2.permute(0, 2, 1)
#         q = self.query(x1)  # Query
#         k = self.key(x2)  # Key
#         v = self.value(x2)  # Value
#         q = q.flatten(start_dim=-2)
#
#         scores = torch.matmul(q.transpose(-2, -1), k)  # Attention scores
#         attention_weights = self.softmax(scores)  # Attention weights
#
#         output = torch.matmul(attention_weights, v.transpose(-2, -1))  # Weighted sum of values
#         return output.transpose(-2, -1)


# # 实例化 CrossAttention1 类
# input_dim = 128  # 假设输入维度为 128
# cross_attention = CrossAttention1(input_dim)
#
# # 创建输入张量
# x1 = torch.randn(1, input_dim, 64, 64)  # 假设 x1 的形状为 (1, 128, 64, 64)
# x2 = torch.randn(1, 4, 128)  # 假设 x2 的形状为 (1, 4, 128)
# x2 = x2.permute(0, 2, 1) # (1, 128, 4)
# print(x1.shape)
# print(x2.shape)
# # 使用 CrossAttention1 进行前向传播
# # c1, c2, c3 = cross_attention(x1, x2)
# # print(c1.shape)
# # print(c2.shape)
# # print(c3.shape)
# output = cross_attention(x1, x2)
# print(output.shape)

# class CrossAttention2(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossAttention2, self).__init__()
#         self.query = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 16, kernel_size=(1, 1), stride=1, bias=False),
#             nn.BatchNorm2d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.key = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.value = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#         x2 = x2.permute(0, 2, 1)
#         q = self.query(x1)  # Query
#         k = self.key(x2)  # Key
#         v = self.value(x2)  # Value
#         q = q.flatten(start_dim=-2)
#
#         scores = torch.matmul(q.transpose(-2, -1), k)  # Attention scores
#         attention_weights = self.softmax(scores)  # Attention weights
#
#         output = torch.matmul(attention_weights, v.transpose(-2, -1))  # Weighted sum of values
#         return output.transpose(-2, -1)
#
# class CrossAttention3(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossAttention3, self).__init__()
#         self.query = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 16, kernel_size=(1, 1), stride=1, bias=False),
#             nn.BatchNorm2d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.key = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.value = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#         x2 = x2.permute(0, 2, 1)
#         q = self.query(x1)  # Query
#         k = self.key(x2)  # Key
#         v = self.value(x2)  # Value
#         q = q.flatten(start_dim=-2)
#
#         scores = torch.matmul(q.transpose(-2, -1), k)  # Attention scores
#         attention_weights = self.softmax(scores)  # Attention weights
#
#         output = torch.matmul(attention_weights, v.transpose(-2, -1))  # Weighted sum of values
#         return output.transpose(-2, -1)
#
# class CrossAttention4(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossAttention4, self).__init__()
#         self.query = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 16, kernel_size=(1, 1), stride=1, bias=False),
#             nn.BatchNorm2d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.key = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.value = nn.Sequential(
#             nn.Conv1d(input_dim, input_dim // 16, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(input_dim // 16),
#             nn.ReLU()
#         )
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#         x2 = x2.permute(0, 2, 1)
#         q = self.query(x1)  # Query
#         k = self.key(x2)  # Key
#         v = self.value(x2)  # Value
#         q = q.flatten(start_dim=-2)
#
#         scores = torch.matmul(q.transpose(-2, -1), k)  # Attention scores
#         attention_weights = self.softmax(scores)  # Attention weights
#
#         output = torch.matmul(attention_weights, v.transpose(-2, -1))  # Weighted sum of values
#         return output.transpose(-2, -1)
import os
for i in range(101):
    folder_name = str(i)
    folder_path = os.path.join('../samples/LEVIR/test/output/', folder_name)
    os.makedirs(folder_path, exist_ok=True)

