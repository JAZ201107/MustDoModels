import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViT(nn.Module):
    def __init__(
        self,
        linear_projection,
        position_embedding,
        encoder,
        mlp_head,
        patch_size=16,
    ):
        super().__init__()
        self.vit = nn.Sequential(
            self.linear_projection,
            self.position_embedding,
            self.encoder,
            self.mlp_head,
        )

        self.patch_size = patch_size

    def forward(self, x):
        nbatchs = x.size(0)
        x = x.reshape(nbatchs, -1, self.patch_size, self.patch_size)

        x = self.vit(x)

        return F.softmax(x, dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        pass


class LinearProjection(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        pass


def attention(key, query, value):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
    ):
        pass


class PositionEmbedding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
    ):
        pass


class MLP(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
    ):
        pass


def make_vit():
    pass
