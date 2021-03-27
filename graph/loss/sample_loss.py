import torch
import torch.nn as nn


class SampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss(reduction='sum')

    def forward(self, recon, x, mu, log_var):
        BCE = self.loss(recon, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD


class MemLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, recon, x, att):
        recon_error = self.loss(recon, x)
        weight_error = torch.sum(att.mul(torch.log(att + 1e-5)))
        return recon_error - (weight_error * 0.0002)