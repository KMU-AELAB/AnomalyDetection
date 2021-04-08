import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from graph.weights_initializer import weights_init


def conv2d(_in):
    return nn.Sequential(
        nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(_in),
        nn.ReLU(inplace=True),
        nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(_in),
        nn.ReLU(inplace=True),
    )


def reduce2d(_in, _out):
    return nn.Sequential(
        nn.Conv2d(_in, _out, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(_out),
        nn.ReLU(inplace=True),
    )


def deconv2d(_in, _out):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=_in, out_channels=_out * 2, kernel_size=4, stride=2,
                           padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=_out * 2, out_channels=_out, kernel_size=4, stride=2,
                           padding=1, bias=False),
        nn.Conv2d(_out, _out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres

        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, query, s):
        score = F.linear(query, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        score = F.softmax(score, dim=1)  # TxM => score

        query = query.contiguous()
        query = query.view(-1, s[1])

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        mem = F.linear(query, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output = torch.cat((query, mem), dim=1)

        return output, self.gather_loss(query, score), self.spread_loss(query, score)

    def gather_loss(self, query, score):
        loss_mse = torch.nn.MSELoss()
        _, gathering_indices = torch.topk(score, 1, dim=1)

        return loss_mse(query, score)

    def spread_loss(self, query, score):
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        _, gathering_indices = torch.topk(score, 2, dim=1)

        pos = self.weight[gathering_indices[:, 0]]
        neg = self.weight[gathering_indices[:, 1]]

        return loss(query, pos.detach(), neg.detach())


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input: torch.Tensor):
        s = input.data.shape
        x = F.normalize(input, p=1, dim=1)
        x = x.permute(0, 2, 3, 1)

        y, g_loss, s_loss = self.memory(x, s)

        y = y.view(s[0], s[2], s[3], s[1])
        y = y.permute(0, 3, 1, 2)

        return y, g_loss, s_loss

class Encoder(nn.Module):
    def __init__(self, width, height, channel, z_channel_size):
        super(Encoder, self).__init__()

        self.conv_lst = nn.ModuleList([
            conv2d(channel),
            conv2d(32),
            conv2d(64),
            conv2d(128),
            conv2d(256),
            conv2d(512),
        ])

        self.reduce_lst = nn.ModuleList([
            reduce2d(channel, 32),
            reduce2d(32, 64),
            reduce2d(64, 128),
            reduce2d(128, 256),
            reduce2d(256, 512),
            reduce2d(512, 1024),
        ])

        self.conv = nn.Conv2d(1024, z_channel_size, kernel_size=1, stride=1, bias=False)

        self.apply(weights_init)

    def forward(self, x):
        for conv, reduce in zip(self.conv_lst, self.reduce_lst):
            _x = conv(x)
            x = x + _x
            x = reduce(x)

        z = self.conv(x)

        return z

class Decoder(nn.Module):
    def __init__(self, width, height, channel, feature_size):
        super(Decoder, self).__init__()

        self.deconv_lst = nn.ModuleList([
            deconv2d(feature_size, 256),
            deconv2d(256, 64),
            deconv2d(64, channel),
        ])

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        for deconv in self.deconv_lst:
            x = deconv(x)

        return self.sigmoid(x)


class Model(nn.Module):
    def __init__(self, width, height, channel=3, z_channel_size=1024, feature_size=1024):
        super(Model, self).__init__()

        self.encoder = Encoder(width, height, channel, z_channel_size)
        self.decoder = Decoder(width, height, channel, feature_size)
        self.mem_module = MemModule(feature_size, z_channel_size)

        self.apply(weights_init)

    def forward(self, x):
        q = self.encoder(x)
        mem_z, g_loss, s_loss = self.mem_module(q)
        out = self.decoder(mem_z)

        return out, g_loss, s_loss
