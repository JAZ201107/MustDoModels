import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import clones


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


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        x = x + self.dropout(layer(self.norm(x)))
        return x


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
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query,
        key,
        value,
    ):
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linrears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.heads * self.d_k)

        del query
        del key
        del value

        return self.linears[-1](x)


class PositionEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Embeddings(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, position_embedding):
        super().__init__()

        self.path_embeddings = position_embedding

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, position_embedding.num_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.path_embeddings(x)

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.layer = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.dropout(self.layer(x).relu())


def make_vit():
    pass
