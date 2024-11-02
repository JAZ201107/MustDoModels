import torch
import torch.nn as nn

import copy


def mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)

    return mask == 0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
