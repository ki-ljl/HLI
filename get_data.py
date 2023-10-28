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


def get_label_graph(path, add_edges, file_name):
    # create label graph
    categories = load_pickle(path + '/data/' + file_name + '/categories.pkl')
    asins = load_pickle(path + '/data/' + file_name + '/asins.pkl')
    #
    layers = []
    for i in range(4):
        temp = []
        for j in range(len(categories)):
            if categories[j][i] not in temp:
                temp.append(categories[j][i])
        layers.append(temp)
    # 1 10 29 82
    all_labels = []
    for x in layers:
        all_labels.extend(x)

    all_labels_dict = dict(zip(all_labels, [x for x in range(len(all_labels))]))

    all_label_to_label = []
    for x in categories:
        for j in range(len(x) - 1):
            start = x[j]
            end = x[j + 1]
            if (all_labels_dict[start], all_labels_dict[end]) not in all_label_to_label:
                all_label_to_label.append((all_labels_dict[start], all_labels_dict[end]))

    label_edge_index = [[], []]
    label_edge_attr = []
    h_adj = torch.zeros(len(all_labels), len(all_labels))
    for x in all_label_to_label:
        label_edge_index[0].append(x[0])
        label_edge_index[1].append(x[1])

        h_adj[x[0], x[1]] = 1
        h_adj[x[1], x[0]] = 1

    ancestors = [[0]]
    label_nums = load_pickle(path + '/data/' + file_name + '/num_labels.pkl')
    total_num = 1 + np.sum(label_nums)
    for i in range(1, total_num):
        temp = [i]
        for k in range(i):
            if (k, i) in all_label_to_label:
                temp.append(k)
        ancestors.append(temp)

    #
    for i in range(1, total_num):
        temp = copy.deepcopy(ancestors[i])
        for j in temp:
            ancestors[i].extend(ancestors[j])

        ancestors[i] = copy.deepcopy(sorted(set(ancestors[i]), key=ancestors[i].index))

    # sibling
    total_num_3 = 1 + label_nums[0] + label_nums[1]
    child_nodes = [[x for x in range(1, label_nums[0] + 1)]]
    for i in range(1, total_num_3):
        temp = []
        for x in all_label_to_label:
            if x[0] == i:
                temp.append(x[1])
        child_nodes.append(temp)

    s_adj = torch.zeros(len(all_labels), len(all_labels))

    if add_edges:
        for x in child_nodes:
            if len(x) == 1:
                continue
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    label_edge_index[0].append(x[i])
                    label_edge_index[1].append(x[j])
                    s_adj[x[i], x[j]] = 1
                    s_adj[x[j], x[i]] = 1

    label_graph_x = load_pickle(path + '/data/' + file_name + '/node2vec_embeddings_64.pkl')
    label_graph_x = torch.Tensor(label_graph_x)
    label_edge_attr = torch.LongTensor(label_edge_attr)
    label_edge_index = torch.LongTensor(label_edge_index)
    label_graph = Data(x=label_graph_x, edge_index=label_edge_index)
    label_graph.edge_index = to_undirected(label_graph.edge_index, num_nodes=label_graph.num_nodes)

    return label_graph, h_adj, s_adj, ancestors

