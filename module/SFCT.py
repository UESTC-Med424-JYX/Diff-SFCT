import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import Convolution
from monai.utils import deprecated_arg


def nonlinearity(x):
    return x * torch.sigmoid(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MultiHeadFrequencyCrossAttention(nn.Module):

    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.to_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.to_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.to_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, q, kv):
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)

        query = rearrange(q, 'b t (d h ) -> b h t d ', h=self.head_num)
        key = rearrange(k, 'b t (d h ) -> b h t d ', h=self.head_num)
        value = rearrange(v, 'b t (d h ) -> b h t d ', h=self.head_num)

        query = torch.fft.fftn(query, dim=(-2, -1))
        key = torch.fft.fftn(key, dim=(-2, -1))

        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        energy = torch.fft.ifftn(energy, dim=(-2, -1)).real

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        out = rearrange(out, "b h t d -> b t (h d)")

        out = self.out_attention(out)

        return out


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class SemanticSpatialTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadSelfAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm1(x)
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x

        x = self.layer_norm2(x)
        _x = self.mlp(x)
        x = x + _x

        return x


class FrequencyCrossTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_cross_attention = MultiHeadFrequencyCrossAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        x = self.layer_norm1(x)
        y = self.layer_norm2(y)

        _x = self.multi_head_cross_attention(x, y)
        _x = self.dropout(_x)
        x = x + _x

        x = self.layer_norm3(x)
        _x = self.mlp(x)
        x = x + _x

        return x


class SpatialFrequencyCrossTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.semantic_spatial_transformer_blocks = nn.ModuleList(
            [SemanticSpatialTransformerBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

        self.frequency_cross_transformer_blocks = nn.ModuleList(
            [FrequencyCrossTransformerBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x, y):
        for sstb, fctb in zip(self.semantic_spatial_transformer_blocks, self.frequency_cross_transformer_blocks):
            y = sstb(y)
            x = fctb(x, y)

        return x


class Spatial_Frequency_Cross_Transformer(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection_x = nn.Linear(self.token_dim, embedding_dim)
        self.embedding_x = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))
        self.dropout_x = nn.Dropout(0.1)

        self.temb_proj = torch.nn.Linear(512, embedding_dim)

        self.projection_y = nn.Linear(self.token_dim, embedding_dim)
        self.embedding_y = nn.Parameter(torch.rand(self.num_tokens, embedding_dim))
        self.dropout_y = nn.Dropout(0.1)

        self.transformer = SpatialFrequencyCrossTransformerBlock(embedding_dim, head_num, mlp_dim, block_num)

    def forward(self, x, y, temb):
        patches_x = rearrange(x,
                              'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                              patch_x=self.patch_dim, patch_y=self.patch_dim)

        patches_y = rearrange(y,
                              'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                              patch_x=self.patch_dim, patch_y=self.patch_dim)

        if temb is not None:
            time_token = self.temb_proj(nonlinearity(temb))
            time_token = time_token.unsqueeze(dim=1)
            patches_x = torch.cat([time_token, patches_x], dim=1)

        batch_size, tokens, _ = patches_x.shape

        patches_x = self.projection_x(patches_x)
        patches_y = self.projection_y(patches_y)

        patches_x += self.embedding_x[:tokens + 1, :]
        patches_y += self.embedding_y[:tokens, :]

        x = self.dropout_x(patches_x)
        y = self.dropout_y(patches_y)

        x = self.transformer(x, y)

        x = x[:, 1:, :]

        x = rearrange(x,
                      ' b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      patch_x=self.patch_dim, patch_y=self.patch_dim, x=self.img_dim)

        return x


if __name__ == '__main__':
    x = torch.rand(1, 1024, 14, 14)
    y = torch.rand(1, 1024, 14, 14)
    temb = torch.rand(1, 512)

    model = Spatial_Frequency_Cross_Transformer(img_dim=14,
                                                in_channels=1024,
                                                embedding_dim=1024,
                                                head_num=4,
                                                mlp_dim=512,
                                                block_num=6,
                                                patch_dim=1)

    out = model(x, y, temb)

    print(out.shape)
