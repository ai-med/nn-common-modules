"""
Description
++++++++++++++++++++++
Building blocks of segmentation neural network

Usage
++++++++++++++++++++++
Import the package and Instantiate any module/block class you want to you::

    from nn_common_modules import modules as additional_modules
    dense_block = additional_modules.DenseBlock(params, se_block_type = 'SSE')

Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.nn.utils import weight_norm


class BayesianConv(nn.Module):
    """Bayesian Convolution

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]

    """

    def __init__(self, params):
        super(BayesianConv, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # conv_out_size = int(params['num_channels'] + params['num_filters'])

        self.conv_mean = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                                   kernel_size=(params['kernel_h'], params['kernel_w']),
                                   padding=(padding_h, padding_w),
                                   stride=params['stride_conv'])

        self.conv_sigma = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                                    kernel_size=(params['kernel_h'], params['kernel_w']),
                                    padding=(padding_h, padding_w),
                                    stride=params['stride_conv'])
        # weights = 0.0001 * torch.ones(
        #     (params['num_filters'], params['num_channels'], params['kernel_h'], params['kernel_w']))
        # bias = 0.0001 * torch.ones(params['num_filters'])
        # self.conv_sigma.weight = nn.Parameter(weights)
        # self.conv_sigma.bias = nn.Parameter(bias)
        # self.conv_mean = weight_norm(self.conv_mean)
        # self.conv_sigma = weight_norm(self.conv_sigma)
        self.normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, switch=False):

        if switch is True:
            x_mean = self.sigmoid(self.conv_mean(input))
            x_sigma = self.sigmoid(self.conv_sigma(torch.mul(input, input)))
            sz = x_sigma.size()
            # TODO: insert Cuda check, Remove harcoded cuda device
            # x_sigma_noise = torch.mul(torch.sqrt(torch.exp(x_sigma)), self.normal.sample(sz).squeeze().cuda())
            x_sigma_noise = torch.mul(torch.sqrt(x_sigma), self.normal.sample(sz).squeeze().cuda())
            out = x_mean + x_sigma_noise
            kl_loss = torch.mean(x_sigma + (x_mean ** 2) - torch.log(x_sigma) - 1)
            # kl_loss = torch.mean(torch.exp(x_sigma + (x_mean ** 2) - x_sigma - 1)
            return out, kl_loss
        else:
            x_mean = self.conv_mean(input)
            return x_mean, None


class EncoderBayesianBlock(nn.Module):
    """
     Encoder Block for Bayesian Network
    """

    def __init__(self, params):
        super(EncoderBayesianBlock, self).__init__()
        self.bayconv = BayesianConv(params)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, pool_required=True, switch=False):
        out, kl_loss = self.bayconv(input, switch)
        out = self.relu(out)
        if pool_required:
            pool, ind = self.maxpool(out)
        else:
            pool, ind = None, None

        return pool, out, ind, kl_loss


class DecoderBayesianBlock(nn.Module):
    """
     Decoder Block for Bayesian Network
    """

    def __init__(self, params):
        super(DecoderBayesianBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])
        self.bayconv = BayesianConv(params)
        self.relu = nn.ReLU()

    def forward(self, input, out_block=None, indices=None, switch=False):
        unpool = self.unpool(input, indices)
        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_conv, kl_loss = self.bayconv(concat, switch)
        out_conv = self.relu(out_conv)
        return out_conv, kl_loss


class ClassifierBayesianBlock(BayesianConv):
    """
    Classifier Bayesian Block
    """

    def __init__(self, params):
        super(ClassifierBayesianBlock, self).__init__(params)
        self.conv_mean = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'],
                                   params['stride_conv'])

        self.conv_sigma = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'],
                                    params['stride_conv'])

    def forward(self, input, switch=False):
        return super().forward(input, switch=switch)
