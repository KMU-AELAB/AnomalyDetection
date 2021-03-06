import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight)

        if m.bias is not None:
            m.weight.data.normal_(-1.0, 1.0)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 1.0)

        if m.bias is not None:
            m.weight.data.normal_(0.0, 1.0)

    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight)

        if m.bias is not None:
            m.weight.data.normal_(-1.0, 1.0)