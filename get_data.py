# -*- coding:utf-8 -*-

import copy
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def get_graph(path, file_name):
    asins = load_pickle(path + '/data/' + file_name + '/asins.pkl')
    also_buys = load_pickle(path + '/data/' + file_name + '/also_buys.pkl')
    categories = load_pickle(path + '/data/' + file_name + '/categories.pkl')
    features = load_pickle(path + '/data/' + file_name + '/hash_vector_embeddings_2048.pkl')

    # dict
    inds = [x for x in range(len(asins))]
    z = dict(zip(asins, inds))

    edge_index = [[], []]
    for i, (asin, also_buy) in enumerate(zip(asins, also_buys)):
        for sub_also_buy in also_buy:
            edge_index[0].append(i)
            edge_index[1].append(z[sub_also_buy])

    layers = []
    for i in range(4):
        temp = []
        for j in range(len(categories)):
            if categories[j][i] not in temp:
                temp.append(categories[j][i])

        layers.append(temp)
    #
    layers_dict = []
    for i, layer in enumerate(layers):
        label_dict = dict(zip(layer, [x for x in range(len(layer))]))
        layers_dict.append(label_dict)
    #
    graph_y = []
    #
    labels_num = [len(layers[x]) for x in range(1, 4)]
    for i, category in enumerate(categories):
        temp_y = []
        for j in range(1, 4):
            x = category[j]
            ind = layers_dict[j][x]
            temp_y.append([ind])
        graph_y.append(temp_y)

    edge_index = torch.LongTensor(edge_index)
    features = torch.Tensor(features)
    labels4 = []
    for label in categories:
        if label[3] not in labels4:
            labels4.append(label[3])

    inds4 = [x for x in range(len(labels4))]
    labels_dict4 = dict(zip(labels4, inds4))
    y = []
    for label in categories:
        y.append(labels_dict4[label[3]])
    graph = Data(x=features, edge_index=edge_index, y=torch.LongTensor(y))

    graph_y = torch.LongTensor(graph_y)

    return graph, graph_y, labels_num


def get_label_graph(path, file_name):
    label_graph = load_pickle(path + '/data/' + file_name + '/label_graph.pkl')
    h_adj = load_pickle(path + '/data/' + file_name + '/h_adj.pkl')
    s_adj = load_pickle(path + '/data/' + file_name + '/s_adj.pkl')
    ancestors = load_pickle(path + '/data/' + file_name + '/ancestors.pkl')

    return label_graph, h_adj, s_adj, ancestors

