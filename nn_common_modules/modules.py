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
from squeeze_and_excitation import squeeze_and_excitation as se
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Block with dense connections

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
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DenseBlock, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class EncoderBlock(DenseBlock):
    """Dense encoder block with maxpool and an optional SE block

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
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor] 
    """

    def __init__(self, params, se_block_type=None):
        super(EncoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, weights=None):
        """Forward pass   
        
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor] 
        """

        out_block = super(EncoderBlock, self).forward(input)
        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(DenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections and SE block

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
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DecoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """
        if indices is not None:
            unpool = self.unpool(input, indices, out_block.shape)
        else:
            # TODO: Implement Conv Transpose
            print("You have to use Conv Transpose")

        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(DecoderBlock, self).forward(concat)

        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


class ClassifierBlock(nn.Module):
    """
    Last layer

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_c':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(
            params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights for classifier regression, defaults to None
        :type weights: torch.tensor (N), optional
        :return: logits
        :rtype: torch.tensor
        """
        batch_size, channel, a, b = input.size()
        if weights is not None:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out_conv = F.conv2d(input, weights)
        else:
            out_conv = self.conv(input)
        return out_conv


class GenericBlock(nn.Module):
    """
    Generic parent class for a conv encoder/decoder block.

    :param params: {'kernel_h': 5
                        'kernel_w': 5
                        'num_channels':64
                        'num_filters':64
                        'stride_conv':1
                        }
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional    
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(GenericBlock, self).__init__()
        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']
        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.prelu = nn.PReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Custom weights for convolution, defaults to None
        :type weights: torch.tensor [FloatTensor], optional
        :return: [description]
        :rtype: [type]
        """

        _, c, h, w = input.shape
        if weights is None:
            x1 = self.conv(input)
        else:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(self.out_channel, c, 1, 1)
            x1 = F.conv2d(input, weights)
        x2 = self.prelu(x1)
        x3 = self.batchnorm(x2)
        return x3


class SDnetEncoderBlock(GenericBlock):
    """
    A standard conv -> prelu -> batchnorm-> maxpool block without dense connections

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
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor] 
    """

    def __init__(self, params, se_block_type=None):
        super(SDnetEncoderBlock, self).__init__(params, se_block_type)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, weights=None):
        """Forward pass   

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]  
        """

        out_block = super(SDnetEncoderBlock, self).forward(input, weights)

        if self.SELayer:
            out_block = self.SELayer(out_block, weights)
        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class SDnetDecoderBlock(GenericBlock):
    """Standard decoder block with maxunpool -> skipconnections -> conv -> prelu -> batchnorm, without dense connections and an optional SE blocks

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
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(SDnetDecoderBlock, self).__init__(params, se_block_type)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: Forward pass
        :rtype: torch.tensor
        """

        unpool = self.unpool(input, indices, out_block.shape)
        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(SDnetDecoderBlock, self).forward(concat, weights)
        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


class SDNetNoBNEncoderBlock(nn.Module):
    """
     Encoder Block for Bayesian Network
    """

    def __init__(self, params):
        super(SDNetNoBNEncoderBlock, self).__init__()
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']
        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.relu(x1)
        out_encoder, indices = self.maxpool(x2)
        return out_encoder, x2, indices


class SDNetNoBNDecoderBlock(nn.Module):
    """
     Decoder Block for Bayesian Network
    """

    def __init__(self, params):
        super(SDNetNoBNDecoderBlock, self).__init__()
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']

        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.relu = nn.ReLU()

        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None):
        unpool = self.unpool(input, indices, out_block.shape)
        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        x1 = self.conv(concat)
        x2 = self.relu(x1)
        return x2
