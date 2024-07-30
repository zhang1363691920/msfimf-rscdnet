import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )
        self.key = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, query_feats, key_feats):
        batch_size = query_feats.size(0)
        channel  =key_feats.size(1)
        query = self.query(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key(key_feats)
        key = key.reshape(*key.shape[:2], -1)

        value = self.value(key_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = (channel ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])

        return context
# cra = CrossAttention1(128)
# input1 = torch.randn(1, 128, 128, 128)
# input2 = torch.randn(1, 128, 4, 1)
# c1 =cra.forward(input1, input2)
# print(c1.shape)











