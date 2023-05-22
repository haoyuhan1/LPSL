import torch
import torch.nn.functional as F
import random
import argparse
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator
from fairness import Fdata, Pdata, PPR, Fdata_relax, Fdata_all, Fdata_sparse, Fdata_sparse1, Fdata_sparse2, Fdata_coordinate, Fdata_block_coordinate, PPR_sparse
import numpy as np
import json
from myutil import sort_trials, Tee
import optuna
import time, datetime
from util import Logger, str2bool
import matplotlib.pyplot as plt
import sys
from model import LP
import math
from torch_geometric.utils import to_networkx, to_edge_index, dense_to_sparse
import networkx as nx
from collections import defaultdict
from model import GCN
from torch_sparse import SparseTensor, matmul
from util import thresholding_operator

def parse_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='Fairness')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')
    parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--const_split', type=str2bool, default=False)
    parser.add_argument('--fix_num', type=int, default=0, help='number of train sample each class')
    parser.add_argument('--defense', type=str, default=None)
    parser.add_argument('--proportion', type=float, default=0, help='proportion of train sample each class')
    parser.add_argument('--prop', type=str, default='APPNP')
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--lambda1', type=float, default=None)
    parser.add_argument('--lambda2', type=float, default=None)
    parser.add_argument('--C', type=float, default=None)
    parser.add_argument('--D', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--Rho', type=float, default=None)
    parser.add_argument('--alr', type=float, default=None, help="learning rate for ADMM")
    parser.add_argument('--sort_key', type=str, default='K')
    parser.add_argument('--LP', type=str2bool, default=False)
    parser.add_argument('--log', type=str2bool, default=True)
    parser.add_argument('--batch_norm', type=str2bool, default=False)
    parser.add_argument('--backbone', type=str, default='MLP')

    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args

class MLP(torch.nn.Module):
    def __init__(self, prop, in_channels, hidden_channels, out_channels, dropout, args, **kwargs):
        super(MLP, self).__init__()
        self.prop = prop
        num_layers = args.num_layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        # self.prop.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data, args, **kwargs):
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if args.batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = torch.mm(self.prop, x)
        return torch.log_softmax(x, dim=-1)
    
    
class GCN1(torch.nn.Module):
    def __init__(self, prop, in_channels, hidden_channels, out_channels, dropout, num_layers, **kwargs):
        super(GCN1, self).__init__()
        self.prop = prop
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.lins:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, args=None, **kwargs):
        # import ipdb; ipdb.set_trace()
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if args.batch_norm:
                x = self.bns[i](x)
            x = torch.mm(self.prop, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = torch.mm(self.prop, x)
        # return x
        return x.log_softmax(dim=-1)
    
    
class LP1(torch.nn.Module):
    def __init__(self, prop, args):
        super().__init__()
        self.prop = prop
    
    def init_label(self, data):
        mask = data.train_mask
        nodes = data.y.shape[0]
        classes = data.y.max() + 1
        label = torch.zeros(nodes, classes)
        label[mask, data.y[mask]] = 1
        label = label.to(data.x.device)
        return label
        
    def forward(self, data, **kwargs):
        x, adj_t, = data.x, data.adj_t
        label = self.init_label(data)
        y = torch.mm(self.prop, label)
        y = torch.clamp(y, 0, 1)
        return y

def train(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    torch.autograd.set_detect_anomaly(True)  ## to locate error of NaN
    optimizer.zero_grad()
    out = model(data, args)
    out = out[train_idx]
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, args=None):
    model.eval()
    out = model(data, args)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc, y_pred


def set_up_trial(trial, args):
    args.lr = trial.suggest_float('lr', 0, 1)
    args.weight_decay = trial.suggest_float('weight_decay', 0, 1)
    args.dropout  = trial.suggest_float('dropout', 0, 1)
    if 'Fairness' in args.model:
        args.lambda1 = trial.suggest_float('lambda1', 0, 100)
        args.lambda2 = trial.suggest_float('lambda2', 0, 100)
        args.C = trial.suggest_float('C', 0, 10)
        args.D = trial.suggest_float('D', 0, 10)
        args.Rho = trial.suggest_float('Rho', 0, 1)
        args.alr = trial.suggest_float('alr', 0, 1)
        args.epsilon = trial.suggest_float('epsilon', 0, 1)
    elif args.model == 'PPR':
        args.alpha = trial.suggest_float('alpha', 0, 1)
    elif args.model == 'Pdata':
        args.lambda1 = trial.suggest_float('lambda1', 0, 100)
        args.alr = trial.suggest_float('alr', 0, 1)
    else:
        raise NotImplementedError
    return args

def set_up_search_space(args):
    dropout_range = [args.dropout]
    lr_range = [args.lr]
    wd_range = [args.weight_decay]
    alpha_range = [args.alpha]
    lambda1_range = [args.lambda1]
    lambda2_range = [args.lambda2]
    C_range = [args.C]
    D_range = [args.D]
    epsilon_range = [args.epsilon]
    Rho_range = [args.Rho]
    alr_range = [args.alr]
    if args.dropout is None:
        dropout_range = [0.5, 0.8]
    if args.lr is None:
        lr_range = [0.001, 0.01, 0.05]
    if args.weight_decay is None:
        wd_range = [5e-4, 5e-5]
    if 'Fairness' in args.model:
        if args.lambda1 is None:
            # lambda1_range = [8, 9, 10, 11, 12, 13]
            lambda1_range = [1, 2, 3, 4, 5, 6, 7, 8]
            # lambda1_range = [1, 2, 3, 4]
            # lambda1_range = [10, 20, 30, 40, 50, 60]
        if args.lambda2 is None:
            lambda2_range = [0.00001, 0.0001, 0.001]
        if args.C is None:
            # C_range = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
            # C_range = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
            C_range = [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
            # C_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        if args.D is None:
            D_range = [0.3, 0.5, 0.7, 1, 1.5]
        if args.Rho is None:
            Rho_range = [0.01, 0.1, 1]
        if args.alr is None:
            alr_range = [0.001, 0.01]
        if args.epsilon is None:
            epsilon_range = [0.1, 0.2, 0.4]
    elif args.model == 'PPR':
        if args.alpha is None:
            alpha_range = [0.1, 0.2]
    elif args.model == 'Pdata':
        if args.lambda1 is None:
            lambda1_range = [60, 70, 80, 90]
            lambda1_range = [10, 20, 30, 40, 50, 60, 70, 80]
        if args.alr is None:
            alr_range = [0.001, 0.01, 0.1]
    search_space = {
        'dropout': dropout_range,
        'lr': lr_range,
        'weight_decay': wd_range,
        'alpha': alpha_range,
        'lambda1': lambda1_range,
        'lambda2': lambda2_range,
        'C': C_range,
        'D': D_range,
        'epsilon': epsilon_range,
        'Rho': Rho_range,
        'alr': alr_range
    }
    return search_space


def main(trial):
    args = parse_args()
    args = set_up_trial(trial, args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')
    logger = Logger(args.runs * random_split_num)
    if args.ogb:
        args.num_layers = 3
        args.weight_decay = 0
        args.hidden_channels = 256

    acc = []
    best = 0
    '''Pdata: get PageRank matrix by gradient descent
    '''
    # lambda1 = 10
    # lr = 0.01
    # Data = Pdata(lambda1, lr, args)
    '''PPR: get Personalized PageRank matrix by iterative method
    '''
    # alpha = 0.1
    # Data = PPR(alpha, args)
    '''Fairness
    '''
    # lambda1 = 10
    # C = 1
    # Rho = 0.01
    # lr = 0.01
    # import ipdb; ipdb.set_trace()
    if args.model == 'Fairness':
        # Data = Fdata(args.lambda1, args.C, args.Rho, args.alr, device, args)
        Data = Fdata_sparse1(args.lambda1, args.C, args.Rho, args.alr, device, args)
    elif args.model == 'Fairness_all':
        Data = Fdata_all(args.lambda1, args.C, args.D, args.Rho, args.alr, device, args)
    elif args.model == 'Fairness2':
        Data = Fdata_relax(args.lambda1, args.C, args.epsilon, args.Rho, args.alr, device, args)
    elif args.model == 'Fairness_sparse':
        Data = Fdata_block_coordinate(args.lambda1, args.lambda2, args.C, args.Rho, args.alr, device, args)
    elif args.model == 'PPR':
        Data = PPR_sparse(args.alpha, args)
    elif args.model == 'Pdata':
        Data = Pdata(args.lambda1, args.alr, args)
    else:
        raise NotImplementedError
    print('current model is', args)
    for split in range(random_split_num):
        P, data, split_idx = Data.process(split=split)
        train_idx = split_idx['train'].to(device)
        print('number of train', len(train_idx))
        other_idx = torch.where(data.train_mask == False)[0]
        args.num_class = data.y.max() + 1
        P, data = P.to(device), data.to(device)
        
        y = data.y.to(device)
                
        if args.backbone == 'LP':
            model = LP1(P, args)
            # model = LP(args)
        elif args.backbone == 'GCN':
            # model = GCN(in_channels=data.num_features, hidden_channels=args.hidden_channels, out_channels=data.num_classes, dropout=args.dropout, num_layers=args.num_layers).to(device)
            model = GCN1(prop=P, in_channels=data.num_features, hidden_channels=args.hidden_channels, out_channels=data.num_classes, dropout=args.dropout, num_layers=args.num_layers).to(device)
        else:
            model = MLP(prop=P, in_channels=data.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=data.num_classes,
                    dropout=args.dropout,
                    args=args).to(device)    
        
        print(model)
        for run in range(args.runs):
            runs_overall = split * args.runs + run
            best_val = 0
            best_test_acc = 0
            if args.LP:
                result = test(model, data, split_idx, args=args)
                train_acc, valid_acc, test_acc, y_pred = result
                logger.add_result(runs_overall, [train_acc, valid_acc, test_acc])
                continue
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            y_best = None
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                loss = train(model, data, train_idx, optimizer, args=args)
                result = test(model, data, split_idx, args=args)
                train_acc, valid_acc, test_acc, y_pred = result
                logger.add_result(runs_overall, [train_acc, valid_acc, test_acc])
                if valid_acc > best_val:
                    best_val = valid_acc
                    best_test_acc = test_acc
                    y_best = y_pred
                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        print(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train Acc: {100 * train_acc:.2f}%, '
                            f'Valid Acc: {100 * valid_acc:.2f}% '
                            f'Test Acc: {100 * test_acc:.2f}%')
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'))
                logger.print_statistics(runs_overall)

    train1_acc, valid_acc, train2_acc, test_acc, \
    train1_var, valid_var, train2_var, test_var = logger.best_result(run=None, with_var=True)
    trial.set_user_attr("train", train2_var)
    trial.set_user_attr("valid", valid_var)
    trial.set_user_attr("test", test_var)

    return valid_acc


if __name__ == '__main__':
    args = parse_args()
    print('args:', args)
    tee = Tee(args)
    if args.log:
        sys.stdout = tee
    search_space = set_up_search_space(args)
    print('search_space:', search_space)
    num_trials = 1
    for s in search_space.values():
        num_trials *= len(s)
    print('num_trials:', num_trials)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(main, n_trials=num_trials)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))
    sorted_trial = sort_trials(study.trials, key=args.sort_key)
    for trial in sorted_trial:
        print("trial.params: ", trial.params, 
              "  trial.value: ", '{0:.5g}'.format(trial.value),
              "  ", trial.user_attrs)
    test_acc = []
    for trial in sorted_trial:
        test_acc.append(trial.user_attrs['test'])
    print('test_acc')
    print(test_acc)

    print("Best params:", study.best_params)
    print("Best trial Value: ", study.best_trial.value)
    print("Best trial Acc: ", study.best_trial.user_attrs)
    tee.close()
    sys.stdout = sys.__stdout__
