'''
Description: 
Author: weihuang
Date: 2021-11-18 15:47:44
LastEditors: weihuang
LastEditTime: 2021-11-22 22:38:26
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):
    def __init__(self, reduction="mean", ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class MSELoss(nn.Module):
    def forward(self,input,target):
        return torch.mean((input-target)**2)

class BCELoss(nn.Module):
    def forward(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
        return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

class WeightedBCELoss(nn.Module):
    def forward(self, input_y, target, weight):
        return F.binary_cross_entropy(input_y, target, weight)

class L1Loss_weighted(nn.Module):
    def forward(self, input, target, weights):
        loss = weights * torch.abs(input - target)
        loss = torch.mean(loss)
        return loss

def weighted_l1_loss(input, target, weights):
    loss = weights * torch.abs(input - target)
    loss = torch.mean(loss)
    return loss

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)