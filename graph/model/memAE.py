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

    def forward(self, x):
        att_weight = F.linear(x, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM

        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = (self.relu(att_weight - self.shrink_thres) * att_weight) /\
                         (torch.abs(att_weight - self.shrink_thres) + 1e-12)
        else:
            self.relu(att_weight)
        att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC

        return output, att_weight  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


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

        x = input.permute(0, 2, 3, 1)

        x = x.contiguous()
        x = x.view(-1, s[1])

        y, att = self.memory(x)

        y = y.view(s[0], s[2], s[3], s[1])
        y = y.permute(0, 3, 1, 2)

        att = att.view(s[0], s[2], s[3], self.mem_dim)
        att = att.permute(0, 3, 1, 2)

        return y, att


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
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        for conv, reduce in zip(self.conv_lst, self.reduce_lst):
            _x = conv(x)
            x = x + _x
            x = reduce(x)

        z = self.lrelu(self.conv(x))

        return z

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


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
        z = self.encoder(x)
        mem_z, att = self.mem_module(z)
        out = self.decoder(mem_z)

        return out, att
