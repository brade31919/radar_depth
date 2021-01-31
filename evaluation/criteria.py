import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class MaskedBerHuLoss(nn.Module):
    def __init__(self, thresh=0.2):
        super(MaskedBerHuLoss, self).__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()

        # Mask out the content
        pred = pred[valid_mask]
        target = target[valid_mask]

        # ipdb.set_trace()
        diff = torch.abs(target - pred)
        delta = self.thresh * torch.max(diff).item()

        part1 = - torch.nn.functional.threshold(-diff, -delta, 0.)
        part2 = torch.nn.functional.threshold(diff ** 2 - delta ** 2, 0., -delta**2.) + delta ** 2
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        loss = torch.mean(loss)

        return loss
