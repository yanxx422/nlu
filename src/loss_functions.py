import torch
import torch.nn as nn

from torch.autograd import Variable


class LossFunction:
    def __init__(self, criterion=nn.CrossEntropyLoss()):
        self.criterion = criterion

    def __call__(self, x, y, norm):
        # baseline: criterion is cross entropy loss
        loss = self.criterion(x, y)
        # experiment: use label smoothing
        # experiment: scale loss as in https://aclanthology.org/E14-1078.pdf
        return loss


# Borrowed from Annotated Transformer -- EXPERIMENT
# TODO: Audit
class LabelSmoothing(nn.Module):

    def __init__(self, n_classes, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))