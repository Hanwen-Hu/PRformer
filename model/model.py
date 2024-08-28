from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor

from .units import SpatialBlock, TemporalBlock, unfold


class Model(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.spatial_blocks = nn.ModuleList([SpatialBlock(args) for _ in range(args.s_block_num)])
        self.season_blocks = nn.ModuleList([TemporalBlock(args, 'season') for _ in range(args.t_block_num)])
        self.trend_blocks = nn.ModuleList([TemporalBlock(args, 'trend') for _ in range(args.t_block_num)])

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (batch, seq_len, dim)
        :return: (batch, pred_len, dim)
        """
        y = torch.mean(x, dim=1, keepdim=True)  # average value
        x = x - y
        for block in self.spatial_blocks:
            x = block(x)
        x = unfold(x, self.args.patch_len, self.args.stride)  # (batch, dim, patch_num, patch_len)
        for s_block, t_block in zip(self.season_blocks, self.trend_blocks):
            y_i, p = s_block(x)
            y = y + y_i
            x = (x - p).detach()
            y_i, p = t_block(x)
            y = y + y_i
            x = (x - p).detach()
        return y
