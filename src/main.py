import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

from torch_geometric.nn import LabelPropagation
from args import args_parser


import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_undirected

from get_data import get_graph, load_pickle, get_label_graph
from models import SAGE, AttGNN
from util import setup_seed, train

setup_seed(1)

args = args_parser()
file_name = args.dataset
device = args.device

graph, graph_y, label_num = get_graph(path=root_path, file_name=file_name)
label_graph, h_adj, s_adj, ancestors = get_label_graph(path=root_path,
                                                       file_name=file_name)

print(graph)
print(label_graph)

h_adj, s_adj = h_adj.to(device), s_adj.to(device)
graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)

if args.experiment == 'full':
    train_mask = load_pickle(root_path + '/data/' + file_name + '/train_mask.pkl')
    val_mask = load_pickle(root_path + '/data/' + file_name + '/val_mask.pkl')
    test_mask = load_pickle(root_path + '/data/' + file_name + '/test_mask.pkl')

else:
    train_mask = load_pickle(root_path + '/data/' + file_name + '/semi_train_mask.pkl')
    val_mask = load_pickle(root_path + '/data/' + file_name + '/semi_val_mask.pkl')
    test_mask = load_pickle(root_path + '/data/' + file_name + '/semi_test_mask.pkl')

graph.train_mask = train_mask
graph.val_mask = val_mask
graph.test_mask = test_mask

print(train_mask.sum())
print(val_mask.sum())
print(test_mask.sum())

graph = graph.to(device)
graph_y = graph_y.to(device)
label_graph = label_graph.to(device)
in_feats = graph.x.shape[1]
label_in_feats = label_graph.x.shape[1]

hidden_feats, out_feats = args.hidden_feats, args.out_feats
num_labels = load_pickle(root_path + '/data/' + file_name + '/num_labels.pkl')

loss_weight = [1, 1, 1]
print(args)


class HLI(nn.Module):
    def __init__(self):
        super(HLI, self).__init__()
        self.label_gnn = AttGNN(label_in_feats, hidden_feats, out_feats,
                                heads=args.num_heads, num_layers=args.num_layers,
                                activation_str='relu')
        self.attr_gnn = SAGE(in_feats, hidden_feats, out_feats)
        self.w12 = nn.Sequential(
            nn.Linear(out_feats, out_feats)
        )
        self.w23 = nn.Sequential(
            nn.Linear(out_feats, out_feats)
        )
        self.alpha_12 = 0.1
        self.alpha_23 = 0.1
        self.alpha = 0.3
        self.beta = 0.2
        self.w = nn.Linear(in_feats, out_feats)
        self.lp = LabelPropagation(num_layers=3, alpha=0.9)
        self.fcs = nn.ModuleList()
        for i in range(3):
            self.fcs.append(nn.Linear(out_feats, num_labels[i]))

    def forward(self):
        x0 = self.w(graph.x)
        x = self.attr_gnn(graph)
        w = self.label_gnn(label_graph, h_adj, s_adj)

        o1 = w[graph_y[:, 0, 0], :]
        o2 = w[graph_y[:, 1, 0], :]
        o3 = w[graph_y[:, 2, 0], :]
        #
        o1 = self.lp(o1, graph.edge_index, mask=graph.train_mask)
        o2 = self.lp(o2, graph.edge_index, mask=graph.train_mask)
        o3 = self.lp(o3, graph.edge_index, mask=graph.train_mask)

        w = w.T

        out0 = torch.matmul(x, w[:, 0])

        out1 = torch.matmul(x, w[:, 1:1 + num_labels[0]])
        out1 = (1 - self.alpha) * out1 + self.alpha * self.fcs[0](o1)

        out12 = self.w12(o1)
        aggr_x_12 = (1 - self.alpha_12 - self.beta) * x + self.alpha_12 * out12 + self.beta * x0
        out2 = torch.matmul(aggr_x_12, w[:, 1 + num_labels[0]:1 + np.sum(num_labels[:2])])
        out2 = (1 - self.alpha) * out2 + self.alpha * self.fcs[1](o2)

        out23 = self.w23(o2)
        aggr_x_23 = (1 - self.alpha_23 - self.beta) * x + self.alpha_23 * out23 + self.beta * x0
        out3 = torch.matmul(aggr_x_23, w[:, 1 + np.sum(num_labels[:2]):1 + np.sum(num_labels)])
        out3 = (1 - self.alpha) * out3 + self.alpha * self.fcs[2](o3)

        out = [out1, out2, out3]

        return out, w.T


def main():
    model = HLI().to(device)
    final_best_acc, final_p, final_r, final_f1, final_jd, final_hl = train(args, model,
                                                                           graph,
                                                                           graph_y,
                                                                           loss_weight,
                                                                           ancestors,
                                                                           num_labels)
    print('best test acc:', final_best_acc[3])
    print('best test f1:', final_f1)
    print('best test jd:', final_jd)
    print('best test hl:', final_hl)


if __name__ == '__main__':
    main()
