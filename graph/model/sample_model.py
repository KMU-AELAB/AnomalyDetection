import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


def conv2d(_in):
    return nn.Sequential(
        nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )

def reduce2d(_in, _out):
    return nn.Sequential(
        nn.Conv2d(_in, _out, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
    )

def deconv2d(_in, _out):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=_in, out_channels=_out * 2, kernel_size=4, stride=2,
                           padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=_out * 2, out_channels=_out, kernel_size=4, stride=2,
                           padding=1, bias=False),
        nn.Conv2d(_out, _out, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_lst = nn.ModuleList([
            conv2d(3),
            conv2d(32),
            conv2d(64),
            conv2d(128),
            conv2d(256),
            conv2d(512),
            conv2d(1024),
        ])

        self.reduce_lst = nn.ModuleList([
            reduce2d(3, 32),
            reduce2d(32, 64),
            reduce2d(64, 128),
            reduce2d(128, 256),
            reduce2d(256, 512),
            reduce2d(512, 1024),
            reduce2d(1024, 1536),
        ])

        self.mu = nn.Conv2d(1536, 1536, kernel_size=3, stride=1, padding=1)
        self.log_var = nn.Conv2d(1536, 1536, kernel_size=3, stride=1, padding=1)

        self.apply(weights_init)

    def forward(self, x):
        for conv, reduce in zip(self.conv_lst, self.reduce_lst):
            _x = conv(x)
            x = x + _x
            x = reduce(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.sampling(mu, log_var), mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv_lst = nn.ModuleList([
            deconv2d(1536, 512),
            deconv2d(512, 128),
            deconv2d(128, 32),
        ])

        self.deconv = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2,
                                         padding=1, bias=False)

        self.mu = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.log_var = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.apply(weights_init)

    def forward(self, x):
        for deconv in self.deconv_lst:
            x = deconv(x)
        x = self.deconv(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.sampling(mu, log_var)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.apply(weights_init)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        out = self.decoder(z)

        return out, mu, log_var
