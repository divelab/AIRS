# adapted from https://github.com/papkov/noise2same.pytorch/blob/d65d5a10f35208f43aed92f3a750025cb9a03a39/noise2same/network.py
# translated from
# https://github.com/divelab/Noise2Same/blob/main/network.py
# https://github.com/divelab/Noise2Same/blob/main/resnet_module.py
from functools import partial
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import normalize
import torch.nn.functional as F
    

#-------------------------------------------

# below is the implementation of UNet adapted from https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py

#-------------------------------------------

import torch
import torch.nn as nn

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet4layers(nn.Module):
    def __init__(self, input_dim=None, output_dim=None):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(input_dim, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ reconstructor """
        self.outputs = nn.Conv2d(64, output_dim, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ reconstructor """
        outputs = self.outputs(d4)

        return outputs
    
class UNet2layers(nn.Module):
    def __init__(self, input_dim=None, output_dim=None):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(input_dim, 64)
        self.e2 = encoder_block(64, 128)

        """ Bottleneck """
        self.b = conv_block(128, 256)

        """ Decoder """
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)

        """ reconstructor """
        self.outputs = nn.Conv2d(64, output_dim, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)

        """ reconstructor """
        outputs = self.outputs(d2)

        return outputs


