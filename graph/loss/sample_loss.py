import torch
import numpy as np
import torch.nn as nn


class SampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, recon, x, mu, log_var):
        BCE = self.loss(recon, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
