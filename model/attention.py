import math
import torch
import torch.nn as nn
from torch import Tensor


def positional_encoding(device: torch.device, max_len: int, hidden_dim: int) -> Tensor:
    pos = torch.arange(max_len, device=device).unsqueeze(1)
    dim = torch.arange(hidden_dim // 2, device=device).unsqueeze(0)
    encoding = torch.zeros(max_len, hidden_dim, device=device)
    encoding[:, 0::2] = torch.sin(2 * torch.pi * pos / max_len ** (2 * dim / hidden_dim))
    encoding[:, 1::2] = torch.cos(2 * torch.pi * pos / max_len ** (2 * dim / hidden_dim))
    return encoding


def score_to_weight(attn_matrix: Tensor) -> Tensor:
    """
    Statistical-driven Weighting Scheme
    :param attn_matrix: (batch, dim, patch_num, patch_num)
    :return:
    """
    attn_shape = attn_matrix.shape
    attn_matrix = attn_matrix.reshape(-1, attn_matrix.shape[-1] * attn_matrix.shape[-2])
    tau = torch.argsort(attn_matrix, dim=-1, descending=True)
    tau = (torch.argsort(tau, dim=-1) + 1) / tau.shape[-1]
    weights = - torch.log(tau).reshape(attn_shape)  # (batch, dim, patch_num, patch_num)
    return weights / torch.sum(weights, dim=-1, keepdim=True)


def season_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    correlation coefficient
    :param query: (batch, dim, patch_num, patch_len)
    :param key: (batch, dim, patch_num, patch_len)
    :param value: (batch, dim, patch_num, hidden_dim)
    :return:
    """
    query = query - query.mean(dim=-1, keepdim=True)
    key = key - key.mean(dim=-1, keepdim=True)
    score = torch.matmul(query, key.transpose(-1, -2))
    q_len = torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True)) + 1e-5  # (batch, dim, patch_num, 1)
    k_len = torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True)) + 1e-5  # (batch, dim, patch_num, 1)
    score = score / torch.matmul(q_len, k_len.transpose(-1, -2))  # (batch, dim, patch_num, patch_num)
    weight = score_to_weight(score)  # weighting
    return torch.matmul(weight, value)


def trend_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    relative distance
    :param query: (batch, dim, patch_num, patch_len)
    :param key: (batch, dim, patch_num, patch_len)
    :param value: (batch, dim, patch_num, hidden_dim)
    :return:
    """
    query = (query - torch.mean(query, dim=-1, keepdim=True)).unsqueeze(-2)  # (batch, dim, patch_num, 1, patch_len)
    key = (key - torch.mean(key, dim=-1, keepdim=True)).unsqueeze(-3)  # (batch, dim, 1, patch_num, patch_len)
    score = -torch.sum((query - key) * (query - key), dim=-1)  # (batch, dim, patch_num, patch_num)
    weight = score_to_weight(score)  # weighting
    return torch.matmul(weight, value)


def spatial_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    representation similarity
    :param query: (batch, dim, hidden_dim)
    :param key: (batch, dim, hidden_dim)
    :param value: (batch, dim, hidden_dim)
    :return:
    """
    score = torch.matmul(query, key.transpose(-1, -2))  # (batch * dim * dim)
    weight = score_to_weight(score)
    return torch.matmul(weight, value)


def normal_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])  # dot product
    weight = torch.softmax(score, dim=-1)
    return torch.matmul(weight, value)


# Pattern-oriented Attention Mechanism
class Attention(nn.Module):
    def __init__(self, device: torch.device, length: int, dim: int, hidden_dim: int, mode: str, dropout: float) -> None:
        super().__init__()
        self.mode = mode
        self.device = device
        self.enc = nn.Sequential(nn.Linear(length, hidden_dim), nn.Dropout(dropout))
        self.dec = nn.Sequential(nn.Linear(hidden_dim, length), nn.Dropout(dropout))
        if mode == 'spatial':
            self.dim_embed = positional_encoding(device, dim, hidden_dim).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Extract Potential Patterns
        :param x: (batch, dim, patch_num, patch_len) || (batch, dim, seq_len)
        :return: (batch, dim, patch_num, patch_len) || (batch, dim, seq_len)
        """
        c = self.enc(x)
        if self.mode == 'spatial':
            c = spatial_attention(c + self.dim_embed, c + self.dim_embed, c)
        elif self.mode == 'season':
            c = season_attention(x, x, c)
        elif self.mode == 'trend':
            c = trend_attention(x, x, c)
        else:
            c = normal_attention(x, x, c)
        return self.dec(c)
