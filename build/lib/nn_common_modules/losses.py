"""
Description
++++++++++++++++++++++
Addition losses module defines classses which are commonly used particularly in segmentation and are not part of standard pytorch library.

Usage
++++++++++++++++++++++
Import the package and Instantiate any loss class you want to you::

    from nn_common_modules import losses as additional_losses
    loss = additional_losses.DiceLoss()

Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable


class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None, binary=False):
        """
        Forward pass

        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return: torch.tensor
        """
        output = F.softmax(output, dim=1)
        if binary:
            return self._dice_loss_binary(output, target)
        return self._dice_loss_multichannel(output, target, weights, ignore_index)

    @staticmethod
    def _dice_loss_binary(output, target):
        """
        Dice loss for one channel binarized input

        :param output: Nx1xHxW logits
        :param target: NxHxW LongTensor
        :return:
        """
        eps = 0.0001

        intersection = output * target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)

    @staticmethod
    def _dice_loss_multichannel(output, target, weights=None, ignore_index=None):
        """
        Forward pass

        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return:
        """
        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class IoULoss(_WeightedLoss):
    """
    IoU Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """Forward pass
        
        :param output: shape = NxCxHxW
        :type output: torch.tensor [FloatTensor]
        :param target: shape = NxHxW
        :type target: torch.tensor [LongTensor]
        :param weights: shape = C, defaults to None
        :type weights: torch.tensor [FloatTensor], optional
        :param ignore_index: index to ignore from loss, defaults to None
        :type ignore_index: int, optional
        :return: loss value
        :rtype: torch.tensor
        """

        output = F.softmax(output, dim=1)

        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1)
        denominator = (output + encoded_target) - (output*encoded_target)

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight)

    def forward(self, inputs, targets):
        """
        Forward pass

        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(inputs, targets)


class CombinedLoss(_Loss):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        input_soft = F.softmax(input, dim=1)
        y_2 = torch.mean(self.dice_loss(input_soft, target))
        if weight is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight.cuda()))
        return y_1 + y_2


# Credit to https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    """

    def __init__(self, gamma=2, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """Forward pass

        :param input: shape = NxCxHxW
        :type input: torch.tensor
        :param target: shape = NxHxW
        :type target: torch.tensor
        :return: loss value
        :rtype: torch.tensor
        """

        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
