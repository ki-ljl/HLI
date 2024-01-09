# -*- coding:utf-8 -*-

import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser(description='hli')

    parser.add_argument('--dataset', type=str, default='IS',
                        help='dataset name', choices=['IS',
                                                      'ACS'])
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--min_epochs', type=int, default=10, help='min training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_feats', type=int, default=512, help='hidden size')
    parser.add_argument('--out_feats', type=int, default=1024, help='out size')
    parser.add_argument('--num_heads', type=int, default=8, help='attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='attention layers')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=100, help='step size')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--gamma', type=float, default=0.2, help='gamma')

    parser.add_argument('--experiment', type=str, default='full',
                        help='experiment type',
                        choices=['full', 'semi'])

    args = parser.parse_args()

    return args
