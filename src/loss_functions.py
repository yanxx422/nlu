import torch
import torch.nn as nn

from torch.autograd import Variable


# experiment: scale loss as in https://aclanthology.org/E14-1078.pdf

# Adapted from Annotated Transformer

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, prediction, target):

        if self.weight is not None:
            prediction = prediction * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(prediction)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * prediction, dim=self.dim))