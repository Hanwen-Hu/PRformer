import matplotlib.pyplot as plt
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_loader import TSDataset
from .model import Model


class PRformer:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        print('Dataset:', args.dataset, '\tPrediction Length:', args.pred_len)
        self.model = Model(args).to(args.device)
        self.file_name = 'files/networks/' + self.args.dataset + str(self.args.pred_len) + '.pth'
        if args.load == 'True':
            state_dict = torch.load(self.file_name)
            self.model.load_state_dict(state_dict)
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = lambda x, y: torch.mean((torch.abs(x - y)))

    def _get_data(self, mode: str) -> DataLoader:
        dataset = TSDataset(self.args.pred_len, self.args.seq_len, self.args.path, self.args.device, mode)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

    def _train_model(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        train_loss = 0
        for x, y in tqdm(loader):
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.mse_func(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        return train_loss / len(loader)

    def _eval_model(self, loader: DataLoader) -> (float, float):
        self.model.eval()
        mse_loss, mae_loss = 0, 0
        with torch.no_grad():
            for x, y in tqdm(loader):
                y_hat = self.model(x)
                mse_loss += self.mse_func(y_hat, y).item()
                mae_loss += self.mae_func(y_hat, y).item()
        return mse_loss / len(loader), mae_loss / len(loader)

    def count_parameter(self) -> None:
        param_num = sum([param.nelement() for param in self.model.parameters()])
        print('Number of Parameters:', param_num)

    def train(self) -> None:
        train_loader = self._get_data('train')
        valid_loader = self._get_data('valid')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        patience, best_valid = 0, float('Inf')
        print('Total Epochs:', self.args.epochs)
        for epoch in range(self.args.epochs):
            train_loss = self._train_model(train_loader, optimizer)
            valid_loss, _ = self._eval_model(valid_loader)
            scheduler.step()
            if valid_loss < best_valid:
                torch.save(self.model.state_dict(), self.file_name)
                best_valid = valid_loss
                patience = 0
            else:
                patience += 1
            print('Epoch', '%02d' % (epoch + 1), 'Train:', round(train_loss, 4), '\tValid:', round(valid_loss, 4), '\tBest:', round(best_valid, 4), '\tPtc:', patience)
            if patience >= self.args.patience:
                print('Early Stop!')
                break

    def test(self) -> None:
        state_dict = torch.load(self.file_name)
        self.model.load_state_dict(state_dict)
        test_loader = self._get_data('test')
        mse_loss, mae_loss = self._eval_model(test_loader)
        print(self.args.dataset, 'Test Loss')
        print('MSE: ', round(mse_loss, 4))
        print('MAE: ', round(mae_loss, 4))

    def visualize(self) -> None:
        dataset = TSDataset(self.args.pred_len, self.args.seq_len, self.args.path, self.args.device, 'test')
        state_dict = torch.load(self.file_name)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        dim_id = int(input('Visual Dimension: '))
        index = int(input('Index: '))
        while index >= 0:
            if index < len(dataset):
                x, y = dataset[index]
                y_bar = self.model(x.unsqueeze(0))
                y_bar = y_bar.squeeze(0)[:, dim_id].detach().cpu().numpy()
                x, y = x[:, dim_id].detach().cpu().numpy(), y[:, dim_id].detach().cpu().numpy()
                plt.figure(figsize=(8, 2.4))
                plt.rcParams['font.sans-serif'] = ['Times New Roman']
                plt.plot(range(self.args.seq_len), x)
                plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), y, label='real')
                plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), y_bar, label='predict')
                plt.legend(fontsize=20)
                plt.yticks([])
                plt.tick_params(labelsize=20)
                plt.xlim([0, self.args.seq_len + self.args.pred_len])
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
                plt.savefig('files/figures/visual_' + self.args.dataset + '.pdf')
                plt.show()
            else:
                print('Out of Range!')
            index = int(input('Index: '))
