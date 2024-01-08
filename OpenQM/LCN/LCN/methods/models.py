import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.random import randn
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch_scatter import scatter_sum

from torch_geometric.utils import from_networkx, degree, sort_edge_index
from torch_geometric.nn import GATConv, GraphConv, GCNConv, GINConv, GINEConv, Set2Set, GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool, LayerNorm, BatchNorm, GlobalAttention
from torch_geometric.data import Data, Batch

class HexagonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(HexagonConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.ones_like(self.conv.weight)
        if mask.shape[-1] == 3:
            row = torch.tensor([[0], [2]])
            col = torch.tensor([[2], [0]])
        elif mask.shape[-1] == 5:
            row = torch.tensor([[0], [0], [1], [-2], [-1], [-1]])
            col = torch.tensor([[-2], [-1], [-1], [0], [0], [1]])
        mask[:,:,row,col] = 0
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class Hexagon108Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(Hexagon108Conv2d, self).__init__()
        self.conv = HexagonConv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=0, padding_mode=padding_mode, bias=bias)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:,:,0,1:8] = x[:,:,12,7:14]
        for i in range(3):
            x[:,:,i,8+i] = x[:,:,6+i,2+i]
        for i in range(4):
            x[:,:,3+i,10+i] = x[:,:,9+i,4+i]
        for i in range(3):
            x[:,:,7+i,13] = x[:,:,1+i,1]
        for i in range(3):
            x[:,:,10+i,14] = x[:,:,4+i,2]
        for i in range(8):
            x[:,:,13,7+i] = x[:,:,1,1+i]
        for i in range(4):
            x[:,:,9+i,3+i] = x[:,:,3+i,9+i]
        for i in range(3):
            x[:,:,6+i,1+i] = x[:,:,0+i,7+i]
        for i in range(2):
            x[:,:,4+i,1] = x[:,:,10+i,13]
        for i in range(4):
            x[:,:,0+i,0] = x[:,:,6+i,12]

        # conv
        x = self.conv(x)

        # postprocessing
        for i in range(5):
            for j in range(8+i, 13):
                x[:,:,i,j] = 0
        for i in range(4):
            x[:,:,2+i,9+i] = 0
        x[:, :, 6:9, 12] = 0

        for i in range(6):
            for j in range(i+1):
                x[:,:,6+i,j] = 0
        for i in range(3):
            x[:,:,5+i,0+i] = 0
        x[:, :, 3:5, 0] = 0

        return x

class Hexagon108RegularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(Hexagon108RegularConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=0, padding_mode=padding_mode, bias=bias)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:,:,0,1:8] = x[:,:,12,7:14]
        for i in range(3):
            x[:,:,i,8+i] = x[:,:,6+i,2+i]
        for i in range(4):
            x[:,:,3+i,10+i] = x[:,:,9+i,4+i]
        for i in range(3):
            x[:,:,7+i,13] = x[:,:,1+i,1]
        for i in range(3):
            x[:,:,10+i,14] = x[:,:,4+i,2]
        for i in range(8):
            x[:,:,13,7+i] = x[:,:,1,1+i]
        for i in range(4):
            x[:,:,9+i,3+i] = x[:,:,3+i,9+i]
        for i in range(3):
            x[:,:,6+i,1+i] = x[:,:,0+i,7+i]
        for i in range(2):
            x[:,:,4+i,1] = x[:,:,10+i,13]
        for i in range(4):
            x[:,:,0+i,0] = x[:,:,6+i,12]

        # conv
        x = self.conv(x)

        # postprocessing
        for i in range(5):
            for j in range(8+i, 13):
                x[:,:,i,j] = 0
        for i in range(4):
            x[:,:,2+i,9+i] = 0
        x[:, :, 6:9, 12] = 0

        for i in range(6):
            for j in range(i+1):
                x[:,:,6+i,j] = 0
        for i in range(3):
            x[:,:,5+i,0+i] = 0
        x[:, :, 3:5, 0] = 0

        return x

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(MaskedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.ones_like(self.conv.weight)
        if mask.shape[-1] == 3:
            row = torch.tensor([[0], [0], [2], [2]])
            col = torch.tensor([[0], [2], [0], [2]])
        elif mask.shape[-1] == 5:
            row = torch.tensor([[0], [0], [0], [0], [1], [1], [3], [3], [4], [4], [4], [4]])
            col = torch.tensor([[0], [1], [3], [4], [0], [4], [0], [4], [0], [1], [3], [4]])
        mask[:,:,row,col] = 0
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(UpConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                  stride=2, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.zeros_like(self.conv.weight)

        row = torch.tensor([[0], [0], [1], [2], [2]])
        col = torch.tensor([[0], [1], [1], [1], [2]])
        mask[:,:,row,col] = 1
        self.register_buffer('mask', mask)

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     with torch.no_grad():
    #         self.conv.weight = nn.Parameter(self.conv.weight * np.sqrt(16 / 5))

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class LeftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(LeftConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                  stride=2, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.zeros_like(self.conv.weight)

        row = torch.tensor([[1], [2], [2], [2], [3]])
        col = torch.tensor([[1], [0], [1], [2], [1]])
        mask[:,:,row,col] = 1
        self.register_buffer('mask', mask)

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     with torch.no_grad():
    #         self.conv.weight = nn.Parameter(self.conv.weight * np.sqrt(16 / 5))

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class RightConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(RightConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                  stride=2, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.zeros_like(self.conv.weight)

        row = torch.tensor([[1], [2], [2], [2], [3]])
        col = torch.tensor([[1], [1], [2], [3], [3]])
        mask[:,:,row,col] = 1
        self.register_buffer('mask', mask)

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     with torch.no_grad():
    #         self.conv.weight = nn.Parameter(self.conv.weight * np.sqrt(16 / 5))

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

## temporary version
class CNN2D(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1)
        for i in range(2):
            self.conv_list.append(nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1))

        self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.num_visible, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])
        self.In_list = nn.ModuleList()
        for i in range(2):
            self.In_list.append(nn.LayerNorm([self.filters, self.height, self.height]))

        self.log_psi = 0
        self.arg_psi = 0

    def psi(self, data, config_in):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        first_out = config
        for i in range(len(self.conv_list)):
            config = self.conv_list[i](config)
            # if (i+1) != len(self.conv_list):
            config = self.In_list[i](config)
            config = F.relu(config)
        # config = self.conv_list[0](config)
        # config = F.relu(config)
        # config = self.conv_list[1](config)

        config = config + first_out
        # config = F.relu(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)
        # config = self.bn3(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)



class CNN2D_SE(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D_SE, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlock, nn.Conv2d, filters, num_blocks=2, height=self.height, stride=1)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.num_visible, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])

        self.Ln = nn.LayerNorm([self.num_visible])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)






class CNN2D_SE_Hex_108(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, num_blocks=2, non_local=False, mode='embedded', conv='pattern'):
        super(CNN2D_SE_Hex_108, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.shape = (12, 13)

        self.conv_list = nn.ModuleList()
        if conv == 'pattern':
            self.first_conv = Hexagon108Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlock, Hexagon108Conv2d, filters, num_blocks=num_blocks, height=self.shape,
                                          stride=1, non_local=non_local, mode=mode)
        elif conv == 'regular':
            self.first_conv = Hexagon108RegularConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=0,padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlock, Hexagon108RegularConv2d, filters, num_blocks=num_blocks, height=self.shape,
                                          stride=1, non_local=non_local, mode=mode)


        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, self.kernel_size, height, stride))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode))
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()

        config = torch.zeros((config_in.shape[0], self.shape[0] * self.shape[1])).cuda()
        cnt = 0
        for i in range(8):
            config[:, self.shape[1] * 0 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 3 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(10):
            config[:, self.shape[1] * 4 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(11):
            config[:, self.shape[1] * 5 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(10):
            config[:, self.shape[1] * 6 + 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 7 + 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 8 + 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 9 + 4 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(8):
            config[:, self.shape[1] * 10 + 5+ i] = config_in[:, cnt]
            cnt += 1
        for i in range(7):
            config[:, self.shape[1] * 11 + 6 + i] = config_in[:, cnt]
            cnt += 1


        config = config.view(-1, 1, self.shape[0], self.shape[1])

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi
        # return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_SE_Hex(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, num_blocks=2, first_kernel_size=3,
                 non_local=False, mode='embedded', preact=False, conv='pattern', sublattice=False, device='cpu', norm='layer', last_conv=False, remove_SE=False):
        super(CNN2D_SE_Hex, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.preact = preact
        self.remove_SE = remove_SE
        self.norm = norm
        self.last_conv = last_conv
        self.sublattice = sublattice
        if self.sublattice:
            self.filters_in = 4
        self.height = int(np.sqrt(self.num_visible))
        self.shape = (self.height, self.height)

        self.conv_list = nn.ModuleList()
        # if first_kernel_size == 3:
        #     self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        # elif first_kernel_size == 5:
        #     self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=5, stride=1, padding=2,
        #                                     padding_mode='circular')
        if conv == 'pattern':
            self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=num_blocks, height=self.shape, stride=1, non_local=non_local, mode=mode)
        elif conv == 'regular':
            if last_conv:
                self.first_conv = nn.Sequential(
                    nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                    nn.LayerNorm([self.filters, self.shape[0], self.shape[1]]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                    nn.LayerNorm([self.filters, self.shape[0], self.shape[1]]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                )
            else:
                self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, padding_mode='circular')

            self.layer1 = self.make_layer(SEResidualBlock, nn.Conv2d, filters, num_blocks=num_blocks, height=self.shape, stride=1, non_local=non_local, mode=mode, preact=self.preact)

        elif conv == 'regular_zero_padding':
            self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=self.kernel_size, stride=1,
                                        padding=self.kernel_size // 2, padding_mode='zeros')
            self.layer1 = self.make_layer(SEResidualBlock, nn.Conv2d, filters, num_blocks=num_blocks, height=self.shape,
                                          stride=1, non_local=non_local, mode=mode)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        if self.last_conv:
            self.final_conv = nn.Sequential(
                nn.Linear(self.filters, self.filters),
                nn.LayerNorm([self.filters]),
                nn.SiLU(inplace=True),
                nn.Linear(self.filters, self.filters),
                nn.LayerNorm([self.filters]),
                nn.SiLU(inplace=True),
                nn.Linear(self.filters, self.filters),
                nn.LayerNorm([self.filters]),
                nn.SiLU(inplace=True),
                nn.Linear(self.filters, 2),
            )
        else:
            self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
            self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        if not self.preact:
            if norm == 'layer':
                self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])
            elif norm == 'batch':
                self.first_In = nn.BatchNorm2d(self.filters)

        if norm == 'layer':
            if self.last_conv:
                self.Ln = nn.LayerNorm([self.filters])
            else:
                self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])
        elif norm == 'batch':
            self.Ln = nn.BatchNorm1d(self.shape[0] * self.shape[1])

        self.log_psi = 0
        self.arg_psi = 0

        if self.sublattice:
            encoding = {}
            l = int(np.sqrt(self.num_visible))
            for i in range(l):
                c = i % 3
                for j in range(l):
                    encoding[i * l + j] = c
                    c = (c + 1) % 3
            ret = torch.tensor([encoding[i] for i in range(self.num_visible)])
            self.sublattice_encoding = torch.nn.functional.one_hot(ret, num_classes=3).view(self.height, self.height, -1).permute(2,0,1).to(device)

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded', preact=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, self.kernel_size, height, stride, preact, self.norm, self.last_conv, remove_SE=self.remove_SE))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode, norm=self.norm))
        # if non_local:
        #     layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode='embedded'))
            # layers.append(block(conv, self.filters, filters, height, stride))

        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        if self.sublattice:
            sublattice_encoding = self.sublattice_encoding.repeat(config.shape[0], 1, 1, 1)
            config = torch.cat((config, sublattice_encoding), dim=1)

        config = self.first_conv(config)
        if not self.preact:
            config = self.first_In(config)
            config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        if self.last_conv:
            config = F.avg_pool2d(config, config.shape[2], config.shape[3]).squeeze() * (config.shape[2] * config.shape[3])
            # config = self.linear1(config)
            # config = self.Ln(config)
            # config = F.relu(config)
            # out = self.linear2(config)  # output log(|psi|) and arg(psi)
            config = self.Ln(config)
            out = self.final_conv(config)
        else:
            config = config.view(config.size(0), -1)
            config = self.linear1(config)
            config = self.Ln(config)
            config = F.relu(config)
            out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi
        # return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)



class SEResidualBlock(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, kernel_size, height, stride=1, preact=False, norm='layer', last_conv=False, remove_SE=False):
        super(SEResidualBlock, self).__init__()

        self.preact = preact
        self.remove_SE = remove_SE

        if self.preact:
            if norm == 'layer':
                if last_conv:
                    self.conv1 = nn.Sequential(
                        nn.LayerNorm([filters_in, height[0], height[1]]),
                        nn.SiLU(inplace=True),
                        conv(filters_in, filters, kernel_size=kernel_size, stride=stride, padding=1, bias=True, padding_mode='circular'),
                    )
                    self.conv2 = nn.Sequential(
                        nn.LayerNorm([filters, height[0], height[1]]),
                        nn.SiLU(inplace=True),
                        conv(filters, filters, kernel_size=kernel_size, stride=1, padding=1, bias=True, padding_mode='circular'),
                    )
                else:
                    self.conv1 = nn.Sequential(
                        nn.LayerNorm([filters_in, height[0], height[1]]),
                        nn.ReLU(inplace=True),
                        conv(filters_in, filters, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True, padding_mode='circular'),
                    )
                    self.conv2 = nn.Sequential(
                        nn.LayerNorm([filters, height[0], height[1]]),
                        nn.ReLU(inplace=True),
                        conv(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True, padding_mode='circular'),
                    )
            elif norm == 'batch':
                self.conv1 = nn.Sequential(
                    nn.BatchNorm2d(filters_in),
                    nn.ReLU(inplace=True),
                    conv(filters_in, filters, kernel_size=kernel_size, stride=stride, padding=1, bias=True, padding_mode='circular'),
                )
                self.conv2 = nn.Sequential(
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    conv(filters, filters, kernel_size=kernel_size, stride=1, padding=1, bias=True, padding_mode='circular'),
                )
        else:
            if norm == 'layer':
                self.conv1 = nn.Sequential(
                    conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                    nn.LayerNorm([filters, height[0], height[1]]),
                    nn.ReLU(inplace=True),
                    )
                self.conv2 = nn.Sequential(
                    conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                    nn.LayerNorm([filters, height[0], height[1]]),
                    # nn.ReLU(inplace=True),
                )
            elif norm == 'batch':
                self.conv1 = nn.Sequential(
                    conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    )
                self.conv2 = nn.Sequential(
                    conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                    nn.BatchNorm2d(filters),
                    # nn.ReLU(inplace=True),
                )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

        if not self.remove_SE:
            self.fc1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
            self.fc2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

        # self.relu = nn.SiLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        if not self.remove_SE:
            # Squeeze
            w = F.avg_pool2d(out, (out.shape[2], out.shape[3]))
            w = F.relu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))
            # Excitation
            out = out * w

        out = out + self.shortcut(x)
        # out = F.relu(out)
        if self.preact:
            return out
        else:
            out = self.relu(out)

        return out

class NonLocalBlock(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1, mode='embedded', act='relu', norm='layer'):
        super(NonLocalBlock, self).__init__()

        self.mode = mode
        self.filters = filters

        self.conv1 = nn.Sequential(
            conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
            # nn.LayerNorm([filters, height[0], height[1]]),
            # nn.ReLU(inplace=True),
            )

        if norm == 'layer':
            self.W_z = nn.Sequential(
                conv(filters, filters_in, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters_in, height[0], height[1]]),
            )
        elif norm == 'batch':
            self.W_z = nn.Sequential(
                conv(filters, filters_in, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.BatchNorm2d(filters_in),
            )

        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        if self.mode == 'embedded' or self.mode == "concatenate":
            self.phi = conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True)
            self.theta = conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True)

        if self.mode == "concatenate":
            if act == 'relu':
                self.W_f = nn.Sequential(
                        nn.Conv2d(in_channels=self.filters * 2, out_channels=1, kernel_size=1),
                        nn.ReLU()
                    )
            elif act == 'swish':
                self.W_f = nn.Sequential(
                        nn.Conv2d(in_channels=self.filters * 2, out_channels=1, kernel_size=1),
                        nn.SiLU(inplace=True)
                    )

        # self.shortcut = nn.Sequential()
        # if stride != 1 or filters_in != filters:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
        #         nn.LayerNorm([filters, height, height])
        #     )


    def forward(self, x):

        batch_size = x.size(0)
        # N C HW
        g = self.conv1(x).view(batch_size, self.filters, -1)
        g = g.permute(0, 2, 1)

        if self.mode == 'embedded':
            theta_x = self.theta(x).view(batch_size, self.filters, -1)
            phi_x = self.phi(x).view(batch_size, self.filters, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.filters, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.filters, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)

        elif self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N

        out = torch.matmul(f_div_C, g)

        # contiguous here just allocates contiguous chunk of memory
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.filters, *x.size()[2:])

        out = self.W_z(out)
        # residual connection
        out = out + x
        # out = F.relu(out)

        return out



class KagomeConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(KagomeConv2D, self).__init__()

        self.up_conv = UpConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.left_conv = LeftConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.right_conv = RightConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 0, 0] = x[:, :, 8, 4]
        x[:, :, 0, 1] = x[:, :, 8, 5]
        x[:, :, 0, 2] = x[:, :, 8, 6]
        x[:, :, 0, 3] = x[:, :, 8, 7]
        x[:, :, 1, 5] = x[:, :, 5, 1]
        x[:, :, 2, 6] = x[:, :, 6, 2]
        x[:, :, 3, 7] = x[:, :, 7, 3]
        x[:, :, 4, 8] = x[:, :, 8, 4]
        x[:, :, 6, 9] = x[:, :, 2, 1]
        x[:, :, 7, 9] = x[:, :, 3, 1]
        x[:, :, 8, 9] = x[:, :, 4, 1]
        x[:, :, 9, 9] = x[:, :, 5, 1]
        x[:, :, 9, 7] = x[:, :, 1, 3]
        x[:, :, 9, 5] = x[:, :, 1, 1]
        x[:, :, 8, 3] = x[:, :, 4, 7]
        x[:, :, 6, 1] = x[:, :, 2, 5]
        x[:, :, 4, 0] = x[:, :, 8, 8]
        x[:, :, 2, 0] = x[:, :, 6, 8]



        up = x.clone()
        left = x.clone()
        right = x.clone()
        up = self.up_conv(up)
        left = self.left_conv(left)
        right = self.right_conv(right)

        zeros = torch.zeros_like(up)

        outputs_up = torch.stack([up.T, zeros.T], dim=1)
        outputs_up = torch.flatten(outputs_up, start_dim=0, end_dim=1).T

        outputs_lr = torch.stack([left.T, right.T], dim=1)
        outputs_lr = torch.flatten(outputs_lr, start_dim=0, end_dim=1).T

        outputs = torch.stack([outputs_up, outputs_lr], dim=3)
        outputs = torch.flatten(outputs, start_dim=2, end_dim=3)


        # postprocessing
        row = torch.tensor([[0], [0], [1], [1], [1], [2], [3], [5], [6], [7], [7], [7]])
        col = torch.tensor([[4], [6], [5], [6], [7], [6], [7], [0], [0], [0], [1], [2]])
        outputs[:,:,row,col] = 0

        return outputs

class Kagome36RegularConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome36RegularConv2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 0, 0] = x[:, :, 8, 4]
        x[:, :, 0, 1] = x[:, :, 8, 5]
        x[:, :, 0, 2] = x[:, :, 8, 6]
        x[:, :, 0, 3] = x[:, :, 8, 7]
        x[:, :, 1, 5] = x[:, :, 5, 1]
        x[:, :, 2, 6] = x[:, :, 6, 2]
        x[:, :, 3, 7] = x[:, :, 7, 3]
        x[:, :, 4, 8] = x[:, :, 8, 4]
        x[:, :, 6, 9] = x[:, :, 2, 1]
        x[:, :, 7, 9] = x[:, :, 3, 1]
        x[:, :, 8, 9] = x[:, :, 4, 1]
        x[:, :, 9, 9] = x[:, :, 5, 1]
        x[:, :, 9, 7] = x[:, :, 1, 3]
        x[:, :, 9, 5] = x[:, :, 1, 1]
        x[:, :, 8, 3] = x[:, :, 4, 7]
        x[:, :, 6, 1] = x[:, :, 2, 5]
        x[:, :, 4, 0] = x[:, :, 8, 8]
        x[:, :, 2, 0] = x[:, :, 6, 8]


        outputs = self.conv(x)

        # postprocessing
        # row = torch.tensor([[0], [0], [1], [1], [1], [2], [3], [5], [6], [7], [7], [7]])
        # col = torch.tensor([[4], [6], [5], [6], [7], [6], [7], [0], [0], [0], [1], [2]])
        # outputs[:,:,row,col] = 0


        return outputs

class Kagome36RegularConv2DV2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome36RegularConv2DV2, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 0, 0] = x[:, :, 8, 4]
        x[:, :, 0, 1] = x[:, :, 8, 5]
        x[:, :, 0, 2] = x[:, :, 8, 6]
        x[:, :, 0, 3] = x[:, :, 8, 7]
        x[:, :, 1, 5] = x[:, :, 5, 1]
        x[:, :, 2, 6] = x[:, :, 6, 2]
        x[:, :, 3, 7] = x[:, :, 7, 3]
        x[:, :, 4, 8] = x[:, :, 8, 4]
        x[:, :, 6, 9] = x[:, :, 2, 1]
        x[:, :, 7, 9] = x[:, :, 3, 1]
        x[:, :, 8, 9] = x[:, :, 4, 1]
        x[:, :, 9, 9] = x[:, :, 5, 1]
        x[:, :, 9, 7] = x[:, :, 1, 3]
        x[:, :, 9, 5] = x[:, :, 1, 1]
        x[:, :, 8, 3] = x[:, :, 4, 7]
        x[:, :, 6, 1] = x[:, :, 2, 5]
        x[:, :, 4, 0] = x[:, :, 8, 8]
        x[:, :, 2, 0] = x[:, :, 6, 8]

        # zero periodic condition padding
        x[:, :, 1, 0] = x[:, :, 5, 8]
        x[:, :, 3, 0] = x[:, :, 7, 8]
        x[:, :, 5, 0] = x[:, :, 1, 4]
        x[:, :, 7, 2] = x[:, :, 3, 6]
        x[:, :, 9, 4] = x[:, :, 5, 8]
        x[:, :, 9, 6] = x[:, :, 1, 2]
        x[:, :, 9, 8] = x[:, :, 1, 4]


        outputs = self.conv(x)

        # postprocessing
        # row = torch.tensor([[0], [0], [1], [1], [1], [2], [3], [5], [6], [7], [7], [7]])
        # col = torch.tensor([[4], [6], [5], [6], [7], [6], [7], [0], [0], [0], [1], [2]])
        # outputs[:,:,row,col] = 0
        for i in range(4):
            for j in range(4 + i, 8):
                outputs[:, :, i, j] = 0
        for i in range(3):
            for j in range(i+1):
                outputs[:, :, 5 + i, j] = 0


        return outputs
    


class Kagome108Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome108Conv2D, self).__init__()

        self.up_conv = UpConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.left_conv = LeftConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.right_conv = RightConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 1, 3] = x[:, :, 13, 15]
        x[:, :, 1, 5] = x[:, :, 13, 5]
        x[:, :, 2, 7] = x[:, :, 14, 7]
        x[:, :, 3, 9] = x[:, :, 15, 9]
        x[:, :, 4, 10] = x[:, :, 16, 10]
        x[:, :, 4, 11] = x[:, :, 16, 11]
        x[:, :, 6, 13] = x[:, :, 6, 1]
        x[:, :, 7, 13] = x[:, :, 7, 1]
        x[:, :, 8, 14] = x[:, :, 8, 2]
        x[:, :, 10, 15] = x[:, :, 10, 3]
        x[:, :, 11, 15] = x[:, :, 11, 3]
        x[:, :, 12, 16] = x[:, :, 12, 4]
        x[:, :, 14, 15] = x[:, :, 2, 3]
        x[:, :, 14, 16] = x[:, :, 2, 4]
        x[:, :, 15, 15] = x[:, :, 3, 3]
        x[:, :, 16, 14] = x[:, :, 4, 2]
        x[:, :, 17, 13] = x[:, :, 5, 1]
        x[:, :, 17, 11] = x[:, :, 5, 11]
        x[:, :, 16, 9] = x[:, :, 4, 9]
        x[:, :, 15, 7] = x[:, :, 3, 7]
        x[:, :, 14, 6] = x[:, :, 2, 6]
        x[:, :, 14, 5] = x[:, :, 2, 5]
        x[:, :, 12, 3] = x[:, :, 12, 15]
        x[:, :, 10, 2] = x[:, :, 10, 14]
        x[:, :, 8, 1] = x[:, :, 8, 13]
        x[:, :, 6, 0] = x[:, :, 6, 12]
        x[:, :, 4, 0] = x[:, :, 16, 12]
        x[:, :, 4, 1] = x[:, :, 16, 13]
        x[:, :, 3, 1] = x[:, :, 15, 13]
        x[:, :, 2, 2] = x[:, :, 14, 14]


        # conv
        up = x.clone()
        left = x.clone()
        right = x.clone()
        up = self.up_conv(up)
        left = self.left_conv(left)
        right = self.right_conv(right)

        zeros = torch.zeros_like(up)

        outputs_up = torch.stack([up.T, zeros.T], dim=1)
        outputs_up = torch.flatten(outputs_up, start_dim=0, end_dim=1).T

        outputs_lr = torch.stack([left.T, right.T], dim=1)
        outputs_lr = torch.flatten(outputs_lr, start_dim=0, end_dim=1).T

        outputs = torch.stack([outputs_up, outputs_lr], dim=3)
        outputs = torch.flatten(outputs, start_dim=2, end_dim=3)


        # postprocessing
        for i in range(9):
            for j in range(7+i, 16):
                outputs[:,:,i,j] = 0


        for i in range(7):
            for j in range(i+1):
                outputs[:,:,9+i,j] = 0

        outputs[:, :, 0, 4:7] = 0
        outputs[:, :, 1, 6:8] = 0
        outputs[:, :, 2, 8] = 0
        outputs[:, :, 3, 9] = 0

        outputs[:, :, 6, 12] = 0
        outputs[:, :, 7, 13] = 0
        outputs[:, :, 8, 14] = 0

        outputs[:, :, 9, 14] = 0
        outputs[:, :, 10, 14] = 0
        outputs[:, :, 11, 15] = 0
        outputs[:, :, 13:, 14:] = 0
        outputs[:, :, 15, 13] = 0

        outputs[:, :, 15, 7:9] = 0
        outputs[:, :, 13, 5] = 0
        outputs[:, :, 14, 6] = 0
        outputs[:, :, 8, 0] = 0
        outputs[:, :, 9, 1] = 0
        outputs[:, :, 7, 0] = 0
        outputs[:, :, 3, 0] = 0
        outputs[:, :, 0:3, 0:2] = 0
        outputs[:, :, 0, 2] = 0

        return outputs

class Kagome108RegularConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome108RegularConv2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 1, 3] = x[:, :, 13, 15]
        x[:, :, 1, 5] = x[:, :, 13, 5]
        x[:, :, 2, 7] = x[:, :, 14, 7]
        x[:, :, 3, 9] = x[:, :, 15, 9]
        x[:, :, 4, 10] = x[:, :, 16, 10]
        x[:, :, 4, 11] = x[:, :, 16, 11]
        x[:, :, 6, 13] = x[:, :, 6, 1]
        x[:, :, 7, 13] = x[:, :, 7, 1]
        x[:, :, 8, 14] = x[:, :, 8, 2]
        x[:, :, 10, 15] = x[:, :, 10, 3]
        x[:, :, 11, 15] = x[:, :, 11, 3]
        x[:, :, 12, 16] = x[:, :, 12, 4]
        x[:, :, 14, 15] = x[:, :, 2, 3]
        x[:, :, 14, 16] = x[:, :, 2, 4]
        x[:, :, 15, 15] = x[:, :, 3, 3]
        x[:, :, 16, 14] = x[:, :, 4, 2]
        x[:, :, 17, 13] = x[:, :, 5, 1]
        x[:, :, 17, 11] = x[:, :, 5, 11]
        x[:, :, 16, 9] = x[:, :, 4, 9]
        x[:, :, 15, 7] = x[:, :, 3, 7]
        x[:, :, 14, 6] = x[:, :, 2, 6]
        x[:, :, 14, 5] = x[:, :, 2, 5]
        x[:, :, 12, 3] = x[:, :, 12, 15]
        x[:, :, 10, 2] = x[:, :, 10, 14]
        x[:, :, 8, 1] = x[:, :, 8, 13]
        x[:, :, 6, 0] = x[:, :, 6, 12]
        x[:, :, 4, 0] = x[:, :, 16, 12]
        x[:, :, 4, 1] = x[:, :, 16, 13]
        x[:, :, 3, 1] = x[:, :, 15, 13]
        x[:, :, 2, 2] = x[:, :, 14, 14]


        # conv
        outputs = self.conv(x)


        return outputs

class Kagome108RegularConv2DV2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome108RegularConv2DV2, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 1, 3] = x[:, :, 13, 15]
        x[:, :, 1, 5] = x[:, :, 13, 5]
        x[:, :, 2, 7] = x[:, :, 14, 7]
        x[:, :, 3, 9] = x[:, :, 15, 9]
        x[:, :, 4, 10] = x[:, :, 16, 10]
        x[:, :, 4, 11] = x[:, :, 16, 11]
        x[:, :, 6, 13] = x[:, :, 6, 1]
        x[:, :, 7, 13] = x[:, :, 7, 1]
        x[:, :, 8, 14] = x[:, :, 8, 2]
        x[:, :, 10, 15] = x[:, :, 10, 3]
        x[:, :, 11, 15] = x[:, :, 11, 3]
        x[:, :, 12, 16] = x[:, :, 12, 4]
        x[:, :, 14, 15] = x[:, :, 2, 3]
        x[:, :, 14, 16] = x[:, :, 2, 4]
        x[:, :, 15, 15] = x[:, :, 3, 3]
        x[:, :, 16, 14] = x[:, :, 4, 2]
        x[:, :, 17, 13] = x[:, :, 5, 1]
        x[:, :, 17, 11] = x[:, :, 5, 11]
        x[:, :, 16, 9] = x[:, :, 4, 9]
        x[:, :, 15, 7] = x[:, :, 3, 7]
        x[:, :, 14, 6] = x[:, :, 2, 6]
        x[:, :, 14, 5] = x[:, :, 2, 5]
        x[:, :, 12, 3] = x[:, :, 12, 15]
        x[:, :, 10, 2] = x[:, :, 10, 14]
        x[:, :, 8, 1] = x[:, :, 8, 13]
        x[:, :, 6, 0] = x[:, :, 6, 12]
        x[:, :, 4, 0] = x[:, :, 16, 12]
        x[:, :, 4, 1] = x[:, :, 16, 13]
        x[:, :, 3, 1] = x[:, :, 15, 13]
        x[:, :, 2, 2] = x[:, :, 14, 14]

        # zero periodic condition padding
        x[:, :, 5, 0] = x[:, :, 5, 12]
        x[:, :, 9, 2] = x[:, :, 9, 14]
        x[:, :, 13, 4] = x[:, :, 1, 4]
        x[:, :, 15, 8] = x[:, :, 3, 8]
        x[:, :, 17, 12] = x[:, :, 5, 12]
        x[:, :, 15, 14] = x[:, :, 3, 2]
        x[:, :, 13, 16] = x[:, :, 1, 4]

        # conv
        outputs = self.conv(x)

        # postprocessing
        for i in range(8):
            for j in range(8+i, 16):
                outputs[:,:,i,j] = 0

        for i in range(7):
            for j in range(i+1):
                outputs[:,:,9+i,j] = 0

        outputs[:, :, 0, 4:8] = 0
        outputs[:, :, 1, 6:9] = 0
        outputs[:, :, 2, 8:10] = 0
        outputs[:, :, 3, 9:11] = 0

        outputs[:, :, 5, 12] = 0
        outputs[:, :, 6, 12:14] = 0
        outputs[:, :, 7, 13:15] = 0
        outputs[:, :, 8, 14:16] = 0

        outputs[:, :, 9, 14:16] = 0
        outputs[:, :, 10, 14:16] = 0
        outputs[:, :, 11, 15] = 0
        outputs[:, :, 12, 15] = 0
        outputs[:, :, 13:, 14:] = 0
        outputs[:, :, 14, 13] = 0
        outputs[:, :, 15, 13] = 0

        outputs[:, :, 15, 7:9] = 0
        outputs[:, :, 13, 5] = 0
        outputs[:, :, 14, 6:8] = 0
        outputs[:, :, 8, 0:2] = 0
        outputs[:, :, 9, 1] = 0
        outputs[:, :, 7, 0] = 0
        outputs[:, :, 3, 0] = 0
        outputs[:, :, 0:3, 0] = 0
        outputs[:, :, 0:2, 0:2] = 0
        outputs[:, :, 0, 2] = 0

        return outputs

class CNN2D_SE_Kagome_108(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, num_blocks=2, non_local=False, mode='embedded',
                 preact=False, conv='pattern', remove_SE=False):
        super(CNN2D_SE_Kagome_108, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.preact = preact
        self.remove_SE = remove_SE
        self.shape = (16,16)

        self.conv_list = nn.ModuleList()

        if conv == 'pattern':
            self.first_conv = Kagome108Conv2D(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1, padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlockKagome, Kagome108Conv2D, filters, num_blocks=num_blocks, height=self.shape, stride=1, non_local=non_local, mode=mode)
        elif conv == 'regular':
            self.first_conv = Kagome108RegularConv2D(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1,
                                              padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlockKagome, Kagome108RegularConv2D, filters, num_blocks=num_blocks,
                                          height=self.shape, stride=1, non_local=non_local, mode=mode)
        elif conv == 'regular-v2':
            self.first_conv = Kagome108RegularConv2DV2(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1,
                                              padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlockKagome, Kagome108RegularConv2DV2, filters, num_blocks=num_blocks,
                                          height=self.shape, stride=1, non_local=non_local, mode=mode)

        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        if not preact:
            self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride, preact=self.preact, remove_SE=self.remove_SE))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode))
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config_in = config_in.clone().float()

        config = torch.zeros((config_in.shape[0], self.shape[0] * self.shape[1])).cuda()
        cnt = 0
        for i in range(4):
            config[:, self.shape[1] * 1 + 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, self.shape[1] * 2 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(8):
            config[:, self.shape[1] * 3 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 4 + 0 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 5 + 0 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 6 + 0 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 7 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 8 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 9 + 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 10 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 11 + 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 12 + 4 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(8):
            config[:, self.shape[1] * 13 + 6 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, self.shape[1] * 14 + 8 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(4):
            config[:, self.shape[1] * 1 + 9 + i] = config_in[:, cnt]
            cnt += 1




        config = config.view(-1, 1, self.shape[0], self.shape[1])
        # config = config.view(-1, 1, 4, 4)
        config = self.first_conv(config)
        if not self.preact:
            config = self.first_In(config)
            config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)


class CNN2D_SE_Kagome(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, num_blocks=2, non_local=False, mode='embedded',
                 preact=False, last_linear_only=False, conv='pattern', sublattice=False, device='cpu', act='relu', norm='layer', last_conv=False, remove_SE=False, only_nonlocal=False):
        super(CNN2D_SE_Kagome, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.preact = preact
        self.norm = norm
        self.last_conv = last_conv
        self.act = act
        self.remove_SE = remove_SE
        self.only_nonlocal = only_nonlocal
        self.sublattice = sublattice
        if self.sublattice:
            self.filters_in = 4
        self.last_linear_only = last_linear_only
        self.shape = (8,8)

        self.conv_list = nn.ModuleList()
        if conv == 'pattern':
            self.first_conv = KagomeConv2D(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1, padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlockKagome, KagomeConv2D, filters, num_blocks=num_blocks, height=self.shape,
                                          stride=1, non_local=non_local, mode=mode, preact=preact)
        elif conv == 'regular':
            if last_conv:
                self.first_conv = nn.Sequential(
                    Kagome36RegularConv2D(self.filters_in, self.filters, kernel_size=3, stride=1, padding=0, padding_mode='circular'),
                    nn.LayerNorm([self.filters, self.shape[0], self.shape[1]]),
                    nn.SiLU(inplace=True),
                    Kagome36RegularConv2D(self.filters, self.filters, kernel_size=3, stride=1, padding=0, padding_mode='circular'),
                    nn.LayerNorm([self.filters, self.shape[0], self.shape[1]]),
                    nn.SiLU(inplace=True),
                    Kagome36RegularConv2D(self.filters, self.filters, kernel_size=3, stride=1, padding=0, padding_mode='circular'),
                )
            else:
                self.first_conv = Kagome36RegularConv2D(self.filters_in, self.filters, kernel_size=3, stride=1, padding=0, padding_mode='circular')

            self.layer1 = self.make_layer(SEResidualBlockKagome, Kagome36RegularConv2D, filters, num_blocks=num_blocks, height=self.shape,
                                          stride=1, non_local=non_local, mode=mode, preact=preact)
        elif conv == 'regular-v2':
            self.first_conv = Kagome36RegularConv2DV2(self.filters_in, self.filters, kernel_size=3, stride=1, padding=0, padding_mode='circular')
            self.layer1 = self.make_layer(SEResidualBlockKagome, Kagome36RegularConv2DV2, filters, num_blocks=num_blocks, height=self.shape,
                                          stride=1, non_local=non_local, mode=mode, preact=preact)
        


        if self.last_conv:
            self.final_conv = nn.Sequential(
                nn.Linear(self.filters, self.filters),
                nn.LayerNorm([self.filters]),
                nn.SiLU(inplace=True),
                nn.Linear(self.filters, self.filters),
                nn.LayerNorm([self.filters]),
                nn.SiLU(inplace=True),
                nn.Linear(self.filters, self.filters),
                nn.LayerNorm([self.filters]),
                nn.SiLU(inplace=True),
                nn.Linear(self.filters, 2),
            )
        else:
            self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
            self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)


        if not self.preact:
            if norm == 'layer':
                self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])
            if norm == 'batch':
                self.first_In = nn.BatchNorm2d(self.filters)

        if norm == 'layer':
            if self.last_conv:
                self.Ln = nn.LayerNorm([self.filters])
            else:
                self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])
        elif norm == 'batch':
            self.Ln = nn.BatchNorm1d(self.shape[0] * self.shape[1])

        # if self.last_linear_only:
        #     self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, 2)
        # else:
        #     self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        #     if norm == 'layer':
        #         self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])
        #     elif norm == 'batch':
        #         self.Ln = nn.BatchNorm1d(self.shape[0] * self.shape[1])
        #     self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.log_psi = 0
        self.arg_psi = 0

        if self.sublattice:
            encoding = {0: 0, 1: 0, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 0, 8: 0, 9: 0, 10: 1, 11: 2, 12: 1, 13: 2, 14: 1,
                        15: 2, 16: 1, 17: 0, 18: 0, 19: 0, 20: 0, 21: 2, 22: 1, 23: 2, 24: 1, 25: 2, 26: 1, 27: 2,
                        28: 0, 29: 0, 30: 0, 31: 2, 32: 1, 33: 2, 34: 1, 35: 2}

            ret = torch.tensor([encoding[i] for i in range(self.num_visible)])
            SLE = torch.nn.functional.one_hot(ret, num_classes=3)

            self.sublattice_encoding = torch.zeros((64, 3))
            cnt = 0
            for i in range(4):
                if i % 2 == 0:
                    self.sublattice_encoding[8 * 0 + i] = SLE[cnt]
                    cnt += 1
            for i in range(5):
                self.sublattice_encoding[8 * 1 + i] = SLE[cnt]
                cnt += 1
            for i in range(6):
                if i % 2 == 0:
                    self.sublattice_encoding[8 * 2 + i] = SLE[cnt]
                    cnt += 1
            for i in range(7):
                self.sublattice_encoding[8 * 3 + i] = SLE[cnt]
                cnt += 1
            for i in range(8):
                if i % 2 == 0:
                    self.sublattice_encoding[8 * 4 + i] = SLE[cnt]
                    cnt += 1
            for i in range(7):
                self.sublattice_encoding[8 * 5 + 1 + i] = SLE[cnt]
                cnt += 1
            for i in range(6):
                if i % 2 == 0:
                    self.sublattice_encoding[8 * 6 + 2 + i] = SLE[cnt]
                    cnt += 1
            for i in range(5):
                self.sublattice_encoding[8 * 7 + 3 + i] = SLE[cnt]
                cnt += 1

            self.sublattice_encoding = self.sublattice_encoding.view(self.shape[0], self.shape[1], -1).permute(2, 0, 1).to(device)



    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded', preact=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if not self.only_nonlocal:
                layers.append(block(conv, self.filters, filters, height, stride, preact, self.act, self.norm, self.last_conv, remove_SE=self.remove_SE)),
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode, norm=self.norm))
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config_in = config_in.clone().float()

        # config = torch.zeros((config_in.shape[0], 48)).cuda()
        # cnt = 0
        # for i in range(8):
        #     if i % 2 == 0:
        #         for j in range(6):
        #             if j % 2 == 0:
        #                 config[:, i * 6 + j] = config_in[:, cnt]
        #                 cnt += 1
        #     else:
        #         for j in range(6):
        #             config[:, i * 6 + j] = config_in[:, cnt]
        #             cnt += 1

        # 2x2x3 kagome input
        # config = torch.zeros((config_in.shape[0], 16)).cuda()
        # cnt = 0
        # for i in range(4):
        #     if i % 2 == 0:
        #         for j in range(4):
        #             if j % 2 == 0:
        #                 config[:, i * 4 + j] = config_in[:, cnt]
        #                 cnt += 1
        #     else:
        #         for j in range(4):
        #             config[:, i * 4 + j] = config_in[:, cnt]
        #             cnt += 1

        config = torch.zeros((config_in.shape[0], 64)).cuda()
        cnt = 0
        for i in range(4):
            if i % 2 == 0:
                config[:, 8 * 0 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(5):
            config[:, 8 * 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, 8 * 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(7):
            config[:, 8 * 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(8):
            if i % 2 == 0:
                config[:, 8 * 4 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(7):
            config[:, 8 * 5 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, 8 * 6 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(5):
            config[:, 8 * 7 + 3 + i] = config_in[:, cnt]
            cnt += 1


        config = config.view(-1, 1, self.shape[0], self.shape[1])

        if self.sublattice:
            sublattice_encoding = self.sublattice_encoding.repeat(config.shape[0], 1, 1, 1)
            config = torch.cat((config, sublattice_encoding), dim=1)

        config = self.first_conv(config)
        if not self.preact:
            config = self.first_In(config)
            if self.act == 'relu':
                config = F.relu(config)
            elif self.act == 'swish':
                config = F.silu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        if self.last_conv:
            config = F.avg_pool2d(config, config.shape[2], config.shape[3]).squeeze() * (config.shape[2] * config.shape[3])
            # config = self.linear1(config)
            # config = self.Ln(config)
            # config = F.relu(config)
            # out = self.linear2(config)  # output log(|psi|) and arg(psi)
            config = self.Ln(config)
            out = self.final_conv(config)
        else:
            config = config.view(config.size(0), -1)
            config = self.linear1(config)
            config = self.Ln(config)
            config = F.relu(config)
            out = self.linear2(config)  # output log(|psi|) and arg(psi)

        # if self.last_linear_only:
        #     out = self.linear1(config)
        # else:
        #     config = self.linear1(config)
        #     config = self.Ln(config)
        #     if self.act == 'relu':
        #         config = F.relu(config)
        #     elif self.act == 'swish':
        #         config = F.silu(config)
        #     # config = F.sigmoid(config)
        #     out = self.linear2(config)  # output log(|psi|) and arg(psi)


        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)



class SEResidualBlockKagome(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1, preact=False, act='relu', norm='layer', last_conv=False, remove_SE=False):
        super(SEResidualBlockKagome, self).__init__()

        self.preact = preact
        self.act = act
        self.remove_SE = remove_SE

        if self.preact:
            if act == 'relu':
                if norm == 'layer':
                    if last_conv:
                        self.conv1 = nn.Sequential(
                            nn.LayerNorm([filters_in, height[0], height[1]]),
                            nn.SiLU(inplace=True),
                            conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True,
                                 padding_mode='circular'),
                        )
                        self.conv2 = nn.Sequential(
                            nn.LayerNorm([filters, height[0], height[1]]),
                            nn.SiLU(inplace=True),
                            conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True,
                                 padding_mode='circular'),
                        )
                    else:
                        self.conv1 = nn.Sequential(
                            nn.LayerNorm([filters_in, height[0], height[1]]),
                            nn.ReLU(inplace=True),
                            conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                        )
                        self.conv2 = nn.Sequential(
                            nn.LayerNorm([filters_in, height[0], height[1]]),
                            nn.ReLU(inplace=True),
                            conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                        )
                elif norm == 'batch':
                    self.conv1 = nn.Sequential(
                        nn.BatchNorm2d(filters_in),
                        nn.ReLU(inplace=True),
                        conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True,
                             padding_mode='circular'),
                    )
                    self.conv2 = nn.Sequential(
                        nn.BatchNorm2d(filters_in),
                        nn.ReLU(inplace=True),
                        conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                    )
            elif act == 'swish':
                self.conv1 = nn.Sequential(
                    nn.LayerNorm([filters, height[0], height[1]]),
                    nn.SiLU(inplace=True),
                    conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                )
                self.conv2 = nn.Sequential(
                    nn.LayerNorm([filters, height[0], height[1]]),
                    nn.SiLU(inplace=True),
                    conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                )
        else:
            if act == 'relu':
                self.conv1 = nn.Sequential(
                    conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                    nn.LayerNorm([filters, height[0], height[1]]),
                    nn.ReLU(inplace=True),
                    )
                self.conv2 = nn.Sequential(
                    conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                    nn.LayerNorm([filters, height[0], height[1]]),
                    # nn.ReLU(inplace=True),
                )
            elif act == 'swish':
                self.conv1 = nn.Sequential(
                    conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                    nn.LayerNorm([filters, height[0], height[1]]),
                    nn.SiLU(inplace=True),
                    )
                self.conv2 = nn.Sequential(
                    conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                    nn.LayerNorm([filters, height[0], height[1]]),
                    # nn.ReLU(inplace=True),
                )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

        if not self.remove_SE:
            self.fc1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
            self.fc2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

        if self.act == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif self.act == 'swish':
            self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        if not self.remove_SE:
            # Squeeze
            w = F.avg_pool2d(out, (out.shape[2], out.shape[3]))
            if self.act == 'relu':
                w = F.relu(self.fc1(w))
            elif self.act == 'swish':
                w = F.silu(self.fc1(w))

            w = torch.sigmoid(self.fc2(w))
            # Excitation
            out = out * w

        out = out + self.shortcut(x)
        # out = F.relu(out)
        if self.preact:
            return out
        else:
            out = self.relu(out)

        return out




class ResidualBlock(torch.nn.Module):
    def __init__(self, filters_in, filters, height, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.LayerNorm([filters, height, height]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm([filters, height, height])
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

    def forward(self, x):
        out = self.conv(x)
        # print(out.shape)
        # print(x.shape)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out



    
    

