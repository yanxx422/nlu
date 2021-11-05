import torch
import torch.nn as nn

from torch.autograd import Variable


# experiment: scale loss as in https://aclanthology.org/E14-1078.pdf

# Adapted from Annotated Transformer

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.reduction = "mean"

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.2, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.reduction = 'mean'

    def forward(self, inputs, targets):
        CE_loss = self.cross_entropy_loss(inputs, targets)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return focal_loss.mean()
