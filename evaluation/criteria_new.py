import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# Define the smoothness loss
class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()
    
    def forward(self, pred_depth, image):
        # Normalize the depth with mean
        depth_mean = pred_depth.mean(2, True).mean(3, True)
        pred_depth_normalized = pred_depth / (depth_mean + 1e-7)

        # Compute the gradient of depth
        grad_depth_x = torch.abs(pred_depth_normalized[:, :, :, :-1] - pred_depth_normalized[:, :, :, 1:])
        grad_depth_y = torch.abs(pred_depth_normalized[:, :, :-1, :] - pred_depth_normalized[:, :, 1:, :])

        # Compute the gradient of the image
        grad_image_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        grad_image_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_image_x)
        grad_depth_y *= torch.exp(-grad_image_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


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


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

    def forward(self, pred, target, mask):
        # pdb.set_trace()
        mask = torch.squeeze(mask.to(torch.bool), dim=-1)
        masked_pred = pred[mask, :]
        masked_target = torch.squeeze(target[mask, :]).to(torch.long)

        return self.cross_entropy(masked_pred, masked_target)
