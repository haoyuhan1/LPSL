from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor, matmul
import numpy as np
import math

from myutil import set_signal_by_label  ## remove

class Propagation(MessagePassing):
    r"""The elastive message passing layer from 
    the paper "Elastic Graph Neural Networks", ICML 2021
    """

    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, 
                 K: int, 
                 mode: str,
                 lambda1: float = None,
                 lambda2: float = None,
                 alpha: float = None,
                 L21: bool = True,
                 dropout: float = 0,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 args = None,
                 **kwargs):

        super(Propagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.mode = mode
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        
        assert add_self_loops == True and normalize == True, ''
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_adj_t = None
        self.args = args
        self.label = None
        self.num_class = args.num_class


    def reset_parameters(self):
        self._cached_adj_t = None
        self.label = None

    def forward(self, x: Tensor, 
                edge_index: Adj, 
                edge_weight: OptTensor = None, 
                data=None,
                FF=None,
                mode=None,
                post_step=None, alpha=None, K=None) -> Tensor:
        """"""
        if self.K <= 0: return x

        assert isinstance(edge_index, SparseTensor), "Only support SparseTensor now"
        assert edge_weight is None, "edge_weight is not supported yet, but it can be extented to weighted case"

        if self.normalize:
            cache = self._cached_adj_t
            if cache is None:
                # edge_index = gcn_norm(  # yapf: disable
                        # edge_index, edge_weight, x.size(self.node_dim), False,
                        # add_self_loops=self.add_self_loops, dtype=x.dtype)
                edge_index = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
                # print(edge_index)
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        hh = x
        mode = self.mode if mode is None else mode
        ## to remove
        # if self.args.LP:
            # x = set_signal_by_label(x, data)
        if alpha is None:
            alpha = self.alpha
        if K is None:
            K = self.K

        if mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K, alpha=alpha)
        
        elif mode == 'ALTOPT':
            x = self.apt_forward(mlp=x, FF=FF, edge_index=edge_index, K=K, alpha=alpha, data=data)
        elif mode == 'CS':
            x = self.label_forward(x=x, edge_index=edge_index, K=self.K, alpha=alpha, post_step=post_step,
                                   edge_weight=edge_weight)
        elif mode == 'ORTGNN':
            x = self.ort_forward(x=x, edge_index=edge_index, K=self.K, alpha=alpha, data=data)

        else:
            raise ValueError('wrong propagate mode')
        return x

    def init_label(self, data, nodes=None, classes=None):
        mask = data.train_mask
        # label = torch.zeros_like(FF).cuda()
        # nodes = data.x.shape[0]
        nodes = data.y.shape[0]
        classes = data.y.max() + 1
        label = torch.zeros(nodes, classes)
        # label = label.cuda()
        # return label
        # label[~mask] = 1/self.num_class
        # import ipdb; ipdb.set_trace()
        # print(label.shape)
        # print(mask, mask.shape)
        # print(data.y[mask], data.y[mask].shape)
        label[mask, data.y[mask]] = 1
        label = label.cuda()
        return label

    def apt_forward(self, mlp, FF, edge_index, K, alpha, data):
        lambda1 = self.args.lambda1
        # lambda1 = self.args.current_epoch / 500
        lambda2 = self.args.lambda2
        # print(lambda1)
        # import ipdb; ipdb.set_trace()
        if not torch.is_tensor(self.label):
            self.label = self.init_label(data)
            print('init label')
        label = self.label
        mask = data.train_mask
        '''
        the mlp weight
        '''
        # weight = 1 - torch.sum(-label * torch.log(mlp.clamp(min=1e-8)), 1) / math.log(self.num_class)
        # indices = (weight < 0.3)
        # mlp[indices] = 1

        if self.args.loss == 'CE':
            # mlp = torch.log_softmax(mlp, dim=-1)
            if self.args.current_epoch != 1:
                mlp = torch.log(mlp)
                # mlp = torch.log_softmax(mlp, dim=-1)
        # print(mlp)
        # print(mlp.max(dim=1)[:20])
        # import ipdb; ipdb.set_trace()
        # mlp = F.softmax(mlp, dim=1)
        for k in range(K):
            AF = self.propagate(edge_index, x=FF, edge_weight=None, size=None)
            if self.args.loss == 'CE':
                FF[mask] = lambda1/(2*(1+lambda2)) * mlp[mask] + 1/(1+lambda2) * AF[mask] + lambda2/(1+lambda2)*label[mask]
                FF[~mask] = lambda1/2 * mlp[~mask] + AF[~mask]
            else:
                FF[mask] = 1/(lambda1+lambda2+1) * AF[mask] + lambda1/(lambda1+lambda2+1) * mlp[mask] + lambda2/(lambda1+lambda2+1) * label[mask]  ## for labeled nodes
                FF[~mask] = 1/(lambda1+1) * AF[~mask] + lambda1/(lambda1+1) * mlp[~mask] ## for unlabeled nodes

        FF = F.softmax(FF/0.2, dim=1)
        # FF = F.softmax(FF / 0.1, dim=1)
        # FF = F.softmax(FF, dim=1)
        # FF = F.softmax(FF/0.5, dim=1)
        return FF

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        for k in range(K):
            Ax = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = alpha * hh + (1 - alpha) * Ax
            # print('now')
            # x = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            # x = x * (1 - alpha)
            # x += alpha * hh
            # x = F.dropout(x, p=self.dropout, training=self.training)
            # x = F.dropout(x, p=self.dropout)
        # import ipdb; ipdb.set_trace()
        # x = F.softmax(x/0.2, dim=1)
        return x

    def label_forward(self, x, edge_index, K, alpha, post_step, edge_weight,):
        out = x
        res = (1-alpha) * out
        for k in range(K):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
            out.mul_(alpha).add_(res)
            if post_step != None:
                out = post_step(out)
        return out

    def ort_forward(self, x, edge_index, K, alpha, data):
        lambda1 = self.args.lambda1
        lambda2 = self.args.lambda2
        # print(lambda1, lambda2, (1+lambda1-2*lambda2))
        # print(lambda1)
        # import ipdb; ipdb.set_trace()
        out = x
        # out = out / out.norm(dim=0, keepdim=True).clamp(min=1e-7)
        # print(out)
        res = 1/(1+lambda1-2*lambda2) * out
        for k in range(K):
            AF = self.propagate(edge_index, x=out, edge_weight=None, size=None)
            FTF = torch.mm(out.T, out)
            FFTF = torch.mm(out, FTF)
            out = lambda1 / (1+lambda1-2*lambda2) * AF + res - 2*lambda2/(1+lambda1-2*lambda2)*FFTF
            # print(FTF)
            # out = out / out.norm(dim=0, keepdim=True).clamp(min=1e-7)
            # print(FTF)
        # out = out / out.norm(dim=0, keepdim=True).clamp(min=1e-7)
        ## 5, 0.01 is good if without norm
        # print(torch.mm(out.T, out))
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, mode={}, lambda1={}, lambda2={}, L21={}, alpha={})'.format(
            self.__class__.__name__, self.K, self.mode, self.lambda1, self.lambda2, self.L21, self.alpha)
