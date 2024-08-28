import json
import argparse
import torch

from model import PRformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Basic Settings
    parser.add_argument('-cuda_id', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-load', type=str, default='False')
    # Dataset Settings
    parser.add_argument('-dataset', type=str, default='ETTh', help='Dataset Name')
    parser.add_argument('-pred_len', type=int, default=24, help='Prediction Length')
    # Model Settings
    parser.add_argument('-s_block_num', type=int, default=2, help='Number of spatial blocks')
    parser.add_argument('-t_block_num', type=int, default=2, help='Number of temporal blocks')
    parser.add_argument('-layer_num', type=int, default=2, help='Number of Layers')
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout Probability')
    args = parser.parse_args()

    with open('configs.json') as file:
        params = json.load(file)[args.dataset]
        args.path = params['path']
        args.dim = params['dim']
        args.seq_len = params['seq_len']
        args.patch_len = params['patch_len']
        args.stride = params['stride']

    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda_id)
    else:
        args.device = torch.device('cpu')
    return args


def main():
    args = parse_args()
    model = PRformer(args)
    model.count_parameter()
    model.train()
    model.test()


if __name__ == '__main__':
    main()
