import torch
import torch.nn as nn

class BiToken(nn.Module):
    def __init__(self):
        super(BiToken, self).__init__()
        self.token_len = 2
        # self.channel_num = [64, 64, 128, 256, 512]  # 示例中的通道数
        self.channel_num = [256, 256, 256, 256, 256]
        self.conv_a = nn.ModuleList([
            nn.Conv2d(self.channel_num[0], self.token_len, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(self.channel_num[1], self.token_len, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(self.channel_num[2], self.token_len, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(self.channel_num[3], self.token_len, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(self.channel_num[4], self.token_len, kernel_size=1, padding=0, bias=False)
        ])
    def _forward_semantic_tokens(self, x, conv_a):
        b, c, h, w = x.shape
        spatial_attention = conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        tokens = tokens.permute(0, 2, 1).contiguous().unsqueeze(3)
        return tokens

    def forward(self, inputs):
        tokens = []
        for x, conv_a in inputs:
            tokens.append(self._forward_semantic_tokens(x, conv_a))
        return tokens

# # 创建模型实例
# model = YourModule()
#
# # 创建输入数据和对应的卷积层列表
# x_list = [output2, output3, output4, output5]  # 示例中的输入列表
# conv_a_list = model.conv_a
#
# # 将输入数据和卷积层作为元组列表传递给forward方法
# inputs = list(zip(x_list, conv_a_list))
#
# # 调用forward方法进行前向传播
# output_tokens = model.forward(inputs)
