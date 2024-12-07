import torch
from torch.utils.data import Dataset
from pandas import read_csv


class TSDataset(Dataset):
    def __init__(self, pred_len: int, seq_len: int, path: str, device: torch.device, mode: str) -> None:
        dataset = read_csv(path)
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.data = torch.Tensor(dataset.iloc[:, 1:].values).to(device)
        self._split(mode)

    def _split(self, mode: str) -> None:
        total_len = self.data.shape[0] - self.seq_len - self.pred_len * 3 + 3
        max_train_idx = int(total_len * 0.7) + self.seq_len + self.pred_len - 1
        max_valid_idx = int(total_len * 0.8) + self.seq_len + self.pred_len * 2 - 2
        avg = torch.mean(self.data, dim=0, keepdim=True)
        std = torch.std(self.data, dim=0, keepdim=True) + 1e-5
        self.data = (self.data - avg) / std
        if mode == 'train':
            self.data = self.data[:max_train_idx]
        elif mode == 'valid':
            self.data = self.data[max_train_idx - self.seq_len: max_valid_idx]
        elif mode == 'test':
            self.data = self.data[max_valid_idx - self.seq_len:]

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        left, mid, right = item, item + self.seq_len, item + self.seq_len + self.pred_len
        x = self.data[left: mid]
        y = self.data[mid: right]
        return x, y
