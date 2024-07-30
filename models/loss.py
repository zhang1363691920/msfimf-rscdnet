import torch
import torch.nn.functional as F
from torch import nn

class CELoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CELoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
            logSoftmax_with_loss
            :param input: torch.Tensor, N*C*H*W
            :param target: torch.Tensor, N*1*H*W,/ N*H*W
            :param weight: torch.Tensor, C
            :return: torch.Tensor [0]
            """
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)