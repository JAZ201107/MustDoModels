import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
