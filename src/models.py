# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)

        return x


def get_activation(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_str == 'elu':
        return nn.ELU()
    elif activation_str == 'prelu':
        return nn.PReLU()
    elif activation_str == 'silu':
        return nn.SiLU()
    elif activation_str == 'gelu':
        return nn.GELU()
    elif activation_str == 'tanh':
        return nn.Tanh()
    elif activation_str == 'softplus':
        return nn.Softplus()
    elif activation_str == 'softsign':
        return nn.Softsign()
    else:
        raise ValueError("Unsupported activation function: " + activation_str)


# HLI
# attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_hiera_att=True, use_sibling_att=True, activation_str='relu'):
        super(AttentionLayer, self).__init__()
        self.use_hiera_att = use_hiera_att
        self.use_sibling_att = use_sibling_att
        self.out_channels = out_channels
        self.beta0 = nn.Parameter(torch.Tensor(3, 1))
        self.beta1 = nn.Parameter(torch.Tensor(3, 1))

        # weight
        self.trans_w = nn.Linear(in_channels, out_channels)
        self.global_att = nn.Linear(2 * out_channels, out_channels)

        self.activation = get_activation(activation_str)

    def get_mask(self, adj):
        upper_mask = torch.triu(torch.ones_like(adj, dtype=torch.bool), diagonal=1)
        diagonal_mask = torch.eye(adj.size(0), dtype=torch.bool)
        lower_mask = torch.tril(torch.ones_like(adj, dtype=torch.bool), diagonal=-1)

        return upper_mask, diagonal_mask, lower_mask

    def get_h_att(self, h, h_adj):
        if self.use_hiera_att:
            e = torch.cosine_similarity(h.detach().unsqueeze(1),
                                        h.detach().unsqueeze(0),
                                        dim=-1)
            upper_mask, diagonal_mask, lower_mask = self.get_mask(e)
            e[upper_mask] = e[upper_mask] * self.beta0[0][0]
            e[diagonal_mask] = e[diagonal_mask] * self.beta0[1][0]
            e[lower_mask] = e[lower_mask] * self.beta0[2][0]
        else:
            e = torch.ones_like(h_adj)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(h_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h = torch.matmul(attention, h)

        return h

    def get_s_att(self, h, s_adj):
        if self.use_sibling_att:
            e = torch.cosine_similarity(h.detach().unsqueeze(1),
                                        h.detach().unsqueeze(0),
                                        dim=-1)
            upper_mask, diagonal_mask, lower_mask = self.get_mask(e)
            e[upper_mask] = e[upper_mask] * self.beta1[0][0]
            e[diagonal_mask] = e[diagonal_mask] * self.beta1[1][0]
            e[lower_mask] = e[lower_mask] * self.beta1[2][0]
        else:
            e = torch.ones_like(s_adj)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(s_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        return h

    def forward(self, x, h_adj, s_adj):
        # 1. trans
        h = self.trans_w(x)
        h1 = self.get_h_att(h, h_adj)
        h2 = self.get_s_att(h, s_adj)
        # concat
        z = torch.cat((h1, h2), dim=-1)
        z = self.global_att(z)

        return z, h


class AttentionConv(nn.Module):
    def __init__(self, in_features, out_features, heads, activation_str='relu'):
        super(AttentionConv, self).__init__()
        self.theta = nn.Linear(in_features, out_features)
        self.M = nn.Linear(2 * out_features, out_features)
        self.attentions = torch.nn.ModuleList()
        for _ in range(heads):
            self.attentions.append(
                AttentionLayer(in_features,
                               out_features,
                               activation_str=activation_str)
            )

    def forward(self, x, h_adj, s_adj):
        zs = []
        hs = []
        for att in self.attentions:
            z, h = att(x, h_adj, s_adj)
            zs.append(z)
            hs.append(h)

        z = torch.mean(torch.stack(zs, dim=0), dim=0)
        h = torch.mean(torch.stack(hs, dim=0), dim=0)

        z1 = self.M(torch.cat([z, h], dim=-1))
        z2 = self.theta(x)

        res = z1 + z2
        res = F.elu(res)

        return res


class AttGNN(torch.nn.Module):
    def __init__(self, num_in,
                 num_hidden,
                 num_out,
                 heads,
                 num_layers,
                 activation_str):
        super(AttGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            AttentionConv(num_in, num_hidden, heads, activation_str=activation_str)
        )
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(num_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(
                AttentionConv(num_in, num_hidden, heads, activation_str=activation_str)
            )
            self.bns.append(nn.BatchNorm1d(num_hidden))
        self.convs.append(
            AttentionConv(num_hidden, num_out, heads, activation_str=activation_str)
        )

    def forward(self, data, h_adj, s_adj):
        x = data.x
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, h_adj, s_adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, h_adj, s_adj)

        return x
