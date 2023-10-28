# -*- coding:utf-8 -*-

import copy
import os
import sys
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pytorchtools import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_total(ys, preds, ancestors, num_labels):
    y, pred = [], []
    labels = [1, 1 + num_labels[0], 1 + np.sum(num_labels[:2])]
    for i, (m, n) in enumerate(zip(ys, preds)):
        m = m.tolist()
        n = n.tolist()
        m = [x + labels[i] for x in m]
        n = [x + labels[i] for x in n]
        y.append(m)
        pred.append(n)

    num_products = len(ys[0])
    sp, sr = 0, 0
    sf1 = 0
    for k in range(num_products):
        p, r = 0, 0
        for i in range(3):
            tr, pr = y[i][k], pred[i][k]
            Ti, Pi = ancestors[tr], ancestors[pr]
            intersection = list(set(Ti) & set(Pi))
            p += len(intersection) / (len(Pi))
            r += len(intersection) / (len(Ti))

        # p += 1   # first layer
        # r += 1
        sp += p / 3
        sr += r / 3

    P = sp / num_products
    R = sr / num_products
    F1 = 2 * P * R / (P + R)

    return P, R, F1


def calculate_jd_and_hl(ys, preds, num_labels):
    y, pred = [], []
    labels = [1, 1 + num_labels[0], 1 + np.sum(num_labels[:2])]
    for i, (m, n) in enumerate(zip(ys, preds)):
        m = m.tolist()
        n = n.tolist()
        m = [x + labels[i] for x in m]
        n = [x + labels[i] for x in n]
        y.append(m)
        pred.append(n)

    s = 0
    t = 0
    for i in range(len(y[0])):
        Y = [y[0][i], y[1][i], y[2][i]]
        P = [pred[0][i], pred[1][i], pred[2][i]]
        s += len(set(Y) & set(P)) / len(set(Y) | set(P))
        t += len(set(Y) ^ set(P)) / 3

    jd = s / len(y[0])
    hl = t / len(y[0])

    return jd, hl


def train(args, model, graph, graph_y, loss_weight,
          ancestors, num_labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    patience = args.patience
    early_stopping = EarlyStopping(patience, verbose=True)
    min_epochs = args.min_epochs
    min_val_loss = np.Inf
    final_best_acc = []
    final_best_p = 0
    final_best_r = 0
    final_best_f1 = 0
    final_best_jd = 0
    final_best_hl = 0
    best_model = None
    model.train()
    train_losses, val_losses = [], []
    for epoch in tqdm(range(args.epochs)):
        out, z = model()
        optimizer.zero_grad()
        total_loss = 0
        y = graph_y[graph.train_mask].to(device)
        for i in range(3):
            total_loss = total_loss + loss_weight[i] * loss_function(out[i][graph.train_mask], y[:, i, :].view(-1))

        total_loss.backward()
        optimizer.step()
        # validation
        val_loss, test_acc, P, R, F1, jd, hl = test(model, graph, graph_y, loss_weight, ancestors, num_labels)
        if val_loss < min_val_loss and epoch + 1 > min_epochs:
            best_model = copy.deepcopy(model)
            min_val_loss = val_loss
            final_best_acc = test_acc
            final_best_p = P
            final_best_r = R
            final_best_f1 = F1
            final_best_jd = jd
            final_best_hl = hl

        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model.train()
        tqdm.write('Epoch{:3d} train_loss {:.5f} val_loss {:.3f}'.
                   format(epoch, total_loss.item(), val_loss))
        train_losses.append(total_loss.item())
        val_losses.append(val_loss)
        print('--------------------------------------------------------------------')
        print('test acc:', test_acc[3])
        print('test Precision:', P)
        print('test Recall:', R)
        print('test F1:', F1)
        print('test jaccard distance:', jd)
        print('test hammling loss:', hl)
        print('--------------------------------------------------------------------')

    # plt.plot(train_losses, c='blue', label='train loss')
    # plt.plot(val_losses, c='red', label='val loss')
    # plt.legend()
    # plt.show()

    return final_best_acc, final_best_p, final_best_r, final_best_f1, final_best_jd, final_best_hl


@torch.no_grad()
def test(model, graph, graph_y, loss_weight, ancestors, num_labels):
    model.eval()
    out, _ = model()
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    total_loss = 0
    y = graph_y[graph.val_mask].to(device)  # [num, 3, 1]
    # print()
    for i in range(3):
        total_loss = total_loss + loss_weight[i] * loss_function(out[i][graph.val_mask], y[:, i, :].view(-1))

    val_loss = total_loss
    # acc
    y = graph_y[graph.test_mask].to(device)  # [num, 3, 1]
    cs = []
    proposed_res = []  #
    preds, ys = [], []
    for i in range(3):
        _, pred = out[i].max(dim=1)

        # F1
        tr = y[:, i, :].cpu().numpy().flatten()
        pr = pred[graph.test_mask].cpu().numpy()
        ys.append(tr)
        preds.append(pr)

        p = pred[graph.test_mask]
        t = y[:, i, :].view(-1)
        proposed_res.append(p.cpu().numpy().tolist())
        proposed_res.append(t.cpu().numpy().tolist())
        cs.append(p.eq(t))
    #
    P, R, F1 = get_total(ys, preds, ancestors, num_labels)
    jd, hl = calculate_jd_and_hl(ys, preds, num_labels)
    #
    acc = []
    for i in range(3):
        acc.append(int(cs[i].sum().item()) / int(graph.test_mask.sum()))
    #
    total = 0
    for i in range(int(graph.test_mask.sum())):
        if cs[0][i] and cs[1][i] and cs[2][i]:
            total += 1
    acc.append(total / int(graph.test_mask.sum()))

    return val_loss.cpu().item(), acc, P, R, F1, jd, hl


def coo2adj(edge_index, num_nodes=None):
    if num_nodes is None:
        return to_scipy_sparse_matrix(edge_index).toarray()
    else:
        return to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).toarray()


def adj2coo(adj):
    """
    adj: numpy
    """
    edge_index_temp = sp.coo_matrix(adj)
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index
