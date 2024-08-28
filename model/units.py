import copy
from argparse import Namespace
from math import sqrt

import torch
import torch.nn as nn
from torch import Tensor

from .attention import Attention, positional_encoding


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)


class Layer(nn.Module):
    def __init__(self, device: torch.device, length: int, dim: int, hidden_dim: int, mode: str, dropout: float = 0) -> None:
        super().__init__()
        self.attn = Attention(device, length, dim, hidden_dim, mode, dropout)
        self.ffn = FeedForward(length, hidden_dim, length, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x) + x
        return self.ffn(x) + x


# Pattern-based Generator
class Generator(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(args.patch_len, args.hidden_dim), nn.Dropout(args.dropout))
        patch_num = (args.seq_len - args.patch_len) // args.stride + 1
        self.pos_embed = positional_encoding(args.device, patch_num, args.hidden_dim).unsqueeze(0).unsqueeze(0)
        self.ffn = FeedForward(args.patch_len, args.hidden_dim, args.pred_len, args.dropout)

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        Derive Representative Patterns
        :param x: (batch, dim, patch_num, patch_len)
        :return: (batch, dim, patch_num, patch_len), (batch, pred_len, dim)
        """
        p = x.detach().clone()
        c = self.enc(x) + self.pos_embed  # (batch, dim, patch_num, hidden_dim)
        score = torch.matmul(c, c.transpose(-1, -2)) / sqrt(c.shape[-1])  # (batch, dim, patch_num, patch_num)
        score = torch.sum(score, dim=-1, keepdim=True)  # (batch, dim, patch_num, 1)
        weight = score / torch.sum(torch.abs(score), dim=-2, keepdim=True)  # (batch, dim, patch_num, 1)
        x = torch.sum(x * weight, dim=-2)  # (batch, dim, patch_len)
        return self.ffn(x).transpose(-1, -2), p


# Seasonal / Trend Block
class TemporalBlock(nn.Module):
    def __init__(self, args: Namespace, mode: str) -> None:
        super().__init__()
        layer = Layer(args.device, args.patch_len, args.dim, args.hidden_dim, mode, args.dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.layer_num)])
        self.generator = Generator(args)

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        :param x: (batch, dim, patch_num, patch_len)
        :return: (batch, pred_len, dim), (batch, dim, patch_num, patch_len)
        """
        for layer in self.layers:
            x = layer(x)
        return self.generator(x)


class SpatialBlock(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        layer = Layer(args.device, args.seq_len, args.dim, args.hidden_dim, 'spatial', args.dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.layer_num)])

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-1, -2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(-1, -2)


# Segmentation
def unfold(x: Tensor, kernel_len: int, stride: int = 1) -> Tensor:
    """
    Slice a tensor into segments
    :param x: (batch, seq_len, dim)
    :param kernel_len: length of segments
    :param stride: distance between segments
    :return: (batch, dim, patch_num, patch_len)
    """
    patch_num = (x.shape[1] - kernel_len) // stride + 1
    sub_len = (patch_num - 1) * stride + kernel_len
    x = x[:, -sub_len:, :].transpose(-1, -2)
    return x.unfold(-1, kernel_len, stride)
