import torch
import torch.nn as nn
from dataset import get_dataset
import argparse
import random
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_dense_adj, add_remaining_self_loops, dense_to_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T
import torch.nn.functional as F
from prop import Propagation
from util import Logger
import optuna
from myutil import sort_trials
from torch_sparse import SparseTensor, matmul
import time
from util import soft_thresholding_operator, thresholding_operator, sparse_masked_row_sum, limit_operator



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--prop', type=str, default='CS')

    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args


class Fdata(object):
    def __init__(self, lambda1, C, Rho, blr, device, args, **kwargs):
        super(Fdata, self).__init__()
        self.lambda1 = lambda1
        self.C = C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        # import ipdb; ipdb.set_trace()
        # adj_t = add_remaining_self_loops(data.adj_t)[0]
        adj = data.adj_t.to_dense()
        adj = adj + torch.eye(adj.shape[0])
        D = torch.sum(adj, dim=1) ** (-0.5)
        D[D==float('inf')] = 0
        D = torch.diag(D)
        adj = torch.mm(torch.mm(D, adj), D)
        L = torch.eye(adj.shape[0]) - adj
        return data, L, train_mask, split_idx

    def update_B(self, B0, y0, L, train_mask):
        # import ipdb; ipdb.set_trace()
        # B = B0.clone()
        B = B0
        I = torch.eye(B.shape[0]).to(self.device)
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([B.shape[0], 1]).float().to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(20):
            B = B - self.lr * (2*(B - I) + 2*self.lambda1*torch.mm(L, B) + torch.mm(y0, mask.T) + self.Rho * torch.mm(torch.mm(B, mask) - self.C * ones, mask.T))
        return B

    def update_y(self, B0, y0, train_mask):
        # y = y0.clone()
        y = y0
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([y.shape[0], 1]).float().to(self.device)
        y = y + self.Rho * (torch.mm(B0, mask) - self.C * ones)
        return y

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        L = L.to(self.device)
        B = torch.eye(data.x.shape[0]).to(self.device)
        y = torch.zeros([data.x.shape[0], 1]).to(self.device)
        train_mask = train_mask.to(self.device)
        import ipdb; ipdb.set_trace()
        for i in range(10):
            B = self.update_B(B, y, L, train_mask)
            y = self.update_y(B, y, train_mask)
        return B, data, split_idx
    
    
class Fdata_sparse1(object):
    def __init__(self, lambda1, C, Rho, blr, device, args, **kwargs):
        super(Fdata_sparse1, self).__init__()
        self.lambda1 = lambda1
        self.C = C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        # self.device = 'cpu'
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        # import ipdb; ipdb.set_trace()
        # adj_t = add_remaining_self_loops(data.adj_t)[0]
        adj = gcn_norm(data.adj_t, add_self_loops=True).to_dense()
        L = torch.eye(adj.shape[0]) - adj
        L = dense_to_sparse(L)
        L = SparseTensor(row=L[0][1], col=L[0][0], value=L[1])
        # L = I - adj
        return data, L, train_mask, split_idx

    def update_B(self, B0, y0, L, train_mask):
        # import ipdb; ipdb.set_trace()
        # B = B0.clone()
        B = B0
        I = torch.eye(B.shape[0]).to(self.device)
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([B.shape[0], 1]).float().to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(20):
            B = B - self.lr * (2*(B - I) + 2*self.lambda1*matmul(L, B) + torch.mm(y0, mask.T) + self.Rho * torch.mm(torch.mm(B, mask) - self.C * ones, mask.T))
            # print(i)
        # print('updata B done')
        return B

    def update_y(self, B0, y0, train_mask):
        # y = y0.clone()
        y = y0
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([y.shape[0], 1]).float().to(self.device)
        y = y + self.Rho * (torch.mm(B0, mask) - self.C * ones)
        return y

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        # import ipdb; ipdb.set_trace()
        L = L.to(self.device)
        B = torch.eye(data.x.shape[0]).to(self.device)
        # B = dense_to_sparse(B)
        # B = SparseTensor(row=B[0][1], col=B[0][0], value=B[1])
        y = torch.zeros([data.x.shape[0], 1]).to(self.device)
        train_mask = train_mask.to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(10):
            # t1 = time.time()
            B = self.update_B(B, y, L, train_mask)
            # print(time.time() - t1)
            y = self.update_y(B, y, train_mask)
        print('update P done')
        return B, data, split_idx
    
class Fdata_sparse2(object):
    def __init__(self, lambda1, lambda2, C, Rho, blr, device, args, **kwargs):
        super(Fdata_sparse2, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C = C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        # self.device = 'cpu'
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        adj = gcn_norm(data.adj_t, add_self_loops=True)
        row, col, value = adj.coo()
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, (data.num_nodes, data.num_nodes))
        self.I = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), torch.ones(data.num_nodes), (data.num_nodes, data.num_nodes))
        L = self.I - adj
        return data, L, train_mask, split_idx

    def update_B(self, B, y, L, train_mask):
        # import ipdb; ipdb.set_trace()
        # B = B0.clone()
        # mask = train_mask.reshape([-1, 1]).float()
        # import ipdb; ipdb.set_trace()
        import ipdb; ipdb.set_trace()
        for i in range(1):
            # B = B - self.lr * (2*(B - self.I) + 2*self.lambda1*torch.mm(L, B) + torch.mm(y, self.mask.T) + self.Rho * torch.mm(torch.mm(B, self.mask) - self.C * self.ones, self.mask.T))
            B = B - self.lr * (2*(B - self.I) + 2*self.lambda1*torch.mm(L, B) + torch.mm(y + self.Rho * (torch.mm(B, self.mask) - self.C * self.ones), self.mask.T))
        B = soft_thresholding_operator(B, self.lambda2 / self.Rho)
        print('updata B')
        print(B)
        import ipdb; ipdb.set_trace()
        # torch.cuda.empty_cache()
        return B

    def update_y(self, B, y, train_mask):
        # mask = train_mask.reshape([-1, 1]).float()
        y = y + self.Rho * (torch.mm(B, self.mask) - self.C * self.ones)
        return y

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        # import ipdb; ipdb.set_trace()
        L = L.to(self.device)
        self.I = self.I.to(self.device)
        B = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), torch.ones(data.num_nodes), (data.num_nodes, data.num_nodes))
        B = B.to(self.device)
        y = torch.zeros([data.x.shape[0], 1])
        y = y.to(self.device)
        y = y.to_sparse_coo()
        mask = train_mask.reshape([-1, 1]).float()
        self.mask = mask.to_sparse_coo()
        self.mask = self.mask.to(self.device)
        ones = torch.ones([y.shape[0], 1]).float()
        ones = ones.to(self.device)
        self.ones = ones.to_sparse_coo()
        # I = torch.eye(B.shape[0]).to(self.device)
        # self.I = I.to_sparse_coo()
        # train_mask = train_mask.to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(10):
            # t1 = time.time()
            B = self.update_B(B, y, L, train_mask)
            # print(time.time() - t1)
            # print(i, B)
            y = self.update_y(B, y, train_mask)
            # print(i, B)
            print(i)
            # print(y)
        import ipdb; ipdb.set_trace()
        row_sum = sparse_masked_row_sum(B, train_mask)
        # B = self.update_B(B, y, L, train_mask)
        # import ipdb; ipdb.set_trace()
        return B, data, split_idx
    
class Fdata_coordinate(object):
    def __init__(self, lambda1, lambda2, C, Rho, blr, device, args, **kwargs):
        super(Fdata_coordinate, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C = C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        # self.device = 'cpu'
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        adj = gcn_norm(data.adj_t, add_self_loops=True)
        row, col, value = adj.coo()
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, (data.num_nodes, data.num_nodes))
        self.I = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), torch.ones(data.num_nodes), (data.num_nodes, data.num_nodes))
        L = self.I - adj
        return data, L, train_mask, split_idx

    def update_B(self, Bs, y, L, train_mask):
        # import ipdb; ipdb.set_trace()
        for i in range(L.shape[0]):
            Ii = self.Is[i]
            if train_mask[i]:
                # B = torch.cat(Bs, dim=1)
                # row_sum = torch.mm(B, self.mask)
                for j in range(1):
                    # import ipdb; ipdb.set_trace()
                    B = torch.cat(Bs, dim=1)
                    Bs[i] = Bs[i] - self.lr * (2*(Bs[i]-Ii) + 2*self.lambda1*torch.mm(L, Bs[i]) + y + self.Rho * (torch.mm(B, self.mask) - self.C * self.ones))
            else:
                for j in range(1):
                    Bs[i] = Bs[i] - self.lr * (2*(Bs[i]-Ii) + 2*self.lambda1*torch.mm(L, Bs[i]))
            Bs[i] = thresholding_operator(Bs[i], self.lambda2 / self.Rho)
            if i % 1 == 0:
                print(i)
        print('updata B')
        return Bs
        
    def update_y(self, Bs, y, train_mask):
        # mask = train_mask.reshape([-1, 1]).float()
        B = torch.cat(Bs, dim=1)
        y = y + self.Rho * (torch.mm(B, self.mask) - self.C * self.ones)
        return y
    
    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        # import ipdb; ipdb.set_trace()
        L = L.to(self.device)
        self.I = self.I.to(self.device)
        n = L.shape[0]
        Bs = [torch.sparse_coo_tensor(indices=torch.tensor([[i], [0]]), values=torch.ones(1), size=(n, 1)).to(self.device) for i in range(n)]
        self.Is = [torch.sparse_coo_tensor(indices=torch.tensor([[i], [0]]), values=torch.ones(1), size=(n, 1)).to(self.device) for i in range(n)]
        y = torch.zeros([data.x.shape[0], 1])
        y = y.to(self.device)
        y = y.to_sparse_coo()
        row_sum = torch.zeros([data.x.shape[0], 1])
        row_sum = row_sum.to(self.device)
        self.row_sum = row_sum.to_sparse_coo()
        mask = train_mask.reshape([-1, 1]).float()
        self.mask = mask.to_sparse_coo()
        self.mask = self.mask.to(self.device)
        B = torch.cat(Bs, dim=1)
        self.row_sum = torch.mm(B, self.mask)
        # import ipdb; ipdb.set_trace()
        ones = torch.ones([y.shape[0], 1]).float()
        ones = ones.to(self.device)
        self.ones = ones.to_sparse_coo()
        for i in range(10):
            B = self.update_B(Bs, y, L, train_mask)
            y = self.update_y(Bs, y, train_mask)
            print(i)
        import ipdb; ipdb.set_trace()
        row_sum = sparse_masked_row_sum(B, train_mask)
        # B = self.update_B(B, y, L, train_mask)
        # import ipdb; ipdb.set_trace()
        return B, data, split_idx
    

class Fdata_block_coordinate(object):
    def __init__(self, lambda1, lambda2, C, Rho, blr, device, args, **kwargs):
        super(Fdata_block_coordinate, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C = C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        # self.device = 'cpu'
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        adj = gcn_norm(data.adj_t, add_self_loops=True)
        row, col, value = adj.coo()
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, (data.num_nodes, data.num_nodes))
        self.I = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), torch.ones(data.num_nodes), (data.num_nodes, data.num_nodes))
        L = self.I - adj
        return data, L, train_mask, split_idx

    def update_B(self, Bs, y, L, train_mask):
        # import ipdb; ipdb.set_trace()
        for i in range(len(Bs)):
            Ii = self.Is[i]
            mask = train_mask[i*Bs[0].shape[1]:i*Bs[0].shape[1]+Bs[i].shape[1]].reshape([-1, 1]).float()
            mask = mask.to_sparse_coo()
            for j in range(10):
                B = torch.cat(Bs, dim=1)
                # Bs[i] = Bs[i] - self.lr * (2*(Bs[i]-Ii) + 2*self.lambda1*torch.mm(L, Bs[i]) + torch.mm(y+self.Rho * (torch.mm(B, self.mask) - self.C * self.ones), mask.T))
                Bs[i] = Bs[i] - self.lr * (2*(Bs[i]-Ii) + 2*self.lambda1*torch.mm(L, Bs[i]))
                if i % 10 == 0 and j == 0:
                    print(i)
                    print(B)
            Bs[i] = soft_thresholding_operator(Bs[i], self.lambda2 / self.Rho)
        print('updata B')
        print(B)
        # import ipdb; ipdb.set_trace()
        return Bs
        
    def update_y(self, Bs, y, train_mask):
        # mask = train_mask.reshape([-1, 1]).float()
        B = torch.cat(Bs, dim=1)
        y = y + self.Rho * (torch.mm(B, self.mask) - self.C * self.ones)
        return y
    
    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        # import ipdb; ipdb.set_trace()
        train_mask = train_mask.to(self.device)
        L = L.to(self.device)
        self.I = self.I.to(self.device)
        n = L.shape[0]
        k = 300  # size of the blocks

        # Create the blocks
        Bs = []
        for i in range(n // k):
            # Create the indices and values for a single block
            block_indices = torch.arange(k)
            block_indices = torch.stack((block_indices + i*k, block_indices))
            block_values = torch.ones(k)
            # Create the block
            block = torch.sparse.FloatTensor(block_indices, block_values, (n, k))
            Bs.append(block)
        remainder = n % k
        if remainder != 0:
            block_indices = torch.arange(remainder)
            block_indices = torch.stack((block_indices + (n // k)*k, block_indices))
            block_values = torch.ones(remainder)
            block = torch.sparse.FloatTensor(block_indices, block_values, (n, remainder))
            Bs.append(block)
        Bs = [B.to(self.device) for B in Bs]
        Is = []
        for i in range(n // k):
            # Create the indices and values for a single block
            block_indices = torch.arange(k)
            block_indices = torch.stack((block_indices + i*k, block_indices))
            block_values = torch.ones(k)
            # Create the block
            block = torch.sparse.FloatTensor(block_indices, block_values, (n, k))
            Is.append(block)
        remainder = n % k
        if remainder != 0:
            block_indices = torch.arange(remainder)
            block_indices = torch.stack((block_indices + (n // k)*k, block_indices))
            block_values = torch.ones(remainder)
            block = torch.sparse.FloatTensor(block_indices, block_values, (n, remainder))
            Is.append(block)
        Is = [I.to(self.device) for I in Is]
        # import ipdb; ipdb.set_trace()
        self.Is = Is
        y = torch.zeros([data.x.shape[0], 1])
        y = y.to(self.device)
        y = y.to_sparse_coo()
        # row_sum = torch.zeros([data.x.shape[0], 1])
        # row_sum = row_sum.to(self.device)
        # self.row_sum = row_sum.to_sparse_coo()
        mask = train_mask.reshape([-1, 1]).float()
        self.mask = mask.to_sparse_coo()
        self.mask = self.mask.to(self.device)
        B = torch.cat(Bs, dim=1)
        self.row_sum = torch.mm(B, self.mask)
        # import ipdb; ipdb.set_trace()
        ones = torch.ones([y.shape[0], 1]).float()
        ones = ones.to(self.device)
        self.ones = ones.to_sparse_coo()
        for i in range(10):
            Bs = self.update_B(Bs, y, L, train_mask)
            y = self.update_y(Bs, y, train_mask)
            print(i)
        import ipdb; ipdb.set_trace()
        B = torch.cat(Bs, dim=1)
        # row_sum = sparse_masked_row_sum(B, train_mask)
        # B = self.update_B(B, y, L, train_mask)
        # import ipdb; ipdb.set_trace()
        return B, data, split_idx


    
class Fdata_sparse(object):
    def __init__(self, lambda1, lambda2, C, Rho, blr, device, args, **kwargs):
        super(Fdata_sparse, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C = C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        # import ipdb; ipdb.set_trace()
        # adj_t = add_remaining_self_loops(data.adj_t)[0]
        adj = data.adj_t.to_dense()
        adj = adj + torch.eye(adj.shape[0])
        D = torch.sum(adj, dim=1) ** (-0.5)
        D[D==float('inf')] = 0
        D = torch.diag(D)
        adj = torch.mm(torch.mm(D, adj), D)
        L = torch.eye(adj.shape[0]) - adj
        return data, L, train_mask, split_idx

    def update_B(self, B, y, Z, U, L, train_mask):
        I = torch.eye(B.shape[0]).to(self.device)
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([B.shape[0], 1]).float().to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(20):
            B = B - self.lr * (2*(B - I) + 2*self.lambda1*torch.mm(L, B) + torch.mm(y, mask.T) + self.Rho * torch.mm(torch.mm(B, mask) - self.C * ones, mask.T) + self.Rho * (B - Z + U))
        return B

    def update_y(self, B, y, train_mask):
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([y.shape[0], 1]).float().to(self.device)
        y = y + self.Rho * (torch.mm(B, mask) - self.C * ones)
        return y
    
    def update_Z(self, B, Z, U):
        # soft thresholding operator
        k = self.lambda2 / self.Rho
        zeros = torch.zeros(B.shape).to(self.device)
        Z = torch.max(B + U - k, zeros) - torch.max(-B - U - k, zeros)
        return Z
    
    def update_U(self, B, Z, U):
        U = U + B - Z
        return U

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        L = L.to(self.device)
        B = torch.eye(data.x.shape[0]).to(self.device)
        Z = torch.zeros([data.x.shape[0], data.x.shape[0]]).to(self.device)
        U = torch.zeros([data.x.shape[0], data.x.shape[0]]).to(self.device)
        y = torch.zeros([data.x.shape[0], 1]).to(self.device)
        train_mask = train_mask.to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(10):
            B = self.update_B(B, y, Z, U, L, train_mask)
            y = self.update_y(B, y, train_mask)
            Z = self.update_Z(B, Z, U)
            U = self.update_U(B, Z, U)
        return B, data, split_idx
    
    
class Fdata_all(object):
    def __init__(self, lambda1, C, D, Rho, blr, device, args, **kwargs):
        super(Fdata_all, self).__init__()
        self.lambda1 = lambda1
        self.C = C
        self.D = D * C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        # import ipdb; ipdb.set_trace()
        # adj_t = add_remaining_self_loops(data.adj_t)[0]
        # adj = data.adj_t.to_dense()
        # adj = adj + torch.eye(adj.shape[0])
        # D = torch.sum(adj, dim=1) ** (-0.5)
        # D[D==float('inf')] = 0
        # D = torch.diag(D)
        # adj = torch.mm(torch.mm(D, adj), D)
        # L = torch.eye(adj.shape[0]) - adj
        adj = gcn_norm(data.adj_t, add_self_loops=True).to_dense()
        L = torch.eye(adj.shape[0]) - adj
        L = dense_to_sparse(L)
        L = SparseTensor(row=L[0][1], col=L[0][0], value=L[1])
        return data, L, train_mask, split_idx

    def update_B(self, B, y1, y2, L, train_mask):
        # import ipdb; ipdb.set_trace()
        I = torch.eye(B.shape[0]).to(self.device)
        mask = train_mask.reshape([-1, 1]).float()
        other_mask = 1 - mask
        ones = torch.ones([B.shape[0], 1]).float().to(self.device)
        # import ipdb; ipdb.set_trace()
        for i in range(20):
        #     B = B - self.lr * (2*(B - I) + 2*self.lambda1*torch.mm(L, B) + torch.mm(y1, mask.T) + self.Rho * torch.mm(torch.mm(B, mask) - self.C * ones, mask.T) \
        #                        + torch.mm(y2, other_mask.T) + self.Rho * torch.mm(torch.mm(B, other_mask)-self.D * ones, other_mask.T))
            B = B - self.lr * (2*(B - I) + 2*self.lambda1*matmul(L, B) + torch.mm(y1, mask.T) + self.Rho * torch.mm(torch.mm(B, mask) - self.C * ones, mask.T) \
                               + torch.mm(y2, other_mask.T) + self.Rho * torch.mm(torch.mm(B, other_mask)-self.D * ones, other_mask.T))
  
        return B

    def update_y1(self, B, y1, train_mask):
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([y1.shape[0], 1]).float().to(self.device)
        y1 = y1 + self.Rho * (torch.mm(B, mask) - self.C * ones)
        return y1
    
    def update_y2(self, B, y2, train_mask):
        mask = 1 - train_mask.reshape([-1, 1]).float()
        ones = torch.ones([y2.shape[0], 1]).float().to(self.device)
        y2 = y2 + self.Rho * (torch.mm(B, mask) - self.D * ones)
        return y2

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        L = L.to(self.device)
        B = torch.eye(data.x.shape[0]).to(self.device)
        y1 = torch.zeros([data.x.shape[0], 1]).to(self.device)
        y2 = torch.zeros([data.x.shape[0], 1]).to(self.device)
        train_mask = train_mask.to(self.device)
        # import ipdb; ipdb.set_trace()
        for _ in range(10):
            B = self.update_B(B, y1, y2, L, train_mask)
            y1 = self.update_y1(B, y1, train_mask)
            y2 = self.update_y2(B, y2, train_mask)
        return B, data, split_idx
    

class Fdata_relax(object):
    def __init__(self, lambda1, C, epsilon, Rho, blr, device, args, **kwargs):
        super(Fdata_relax, self).__init__()
        self.lambda1 = lambda1
        self.C = C
        self.epsilon = epsilon * C # epsilon * C
        self.Rho = Rho
        self.lr = blr
        self.device = device
        self.args = args

    def get_data(self, split=0):
        args = self.args
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        # import ipdb; ipdb.set_trace()
        # adj_t = add_remaining_self_loops(data.adj_t)[0]
        adj = data.adj_t.to_dense()
        adj = adj + torch.eye(adj.shape[0])
        D = torch.sum(adj, dim=1) ** (-0.5)
        D[D==float('inf')] = 0
        D = torch.diag(D)
        adj = torch.mm(torch.mm(D, adj), D)
        L = torch.eye(adj.shape[0]) - adj
        return data, L, train_mask, split_idx
    
    def update_z(self, z1, y1, z2, y2, B, train_mask):
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([B.shape[0], 1]).float().to(self.device)
        for _ in range(50):
            z1 = z1 - self.lr * (2*y1*z1 + 2*self.Rho*(torch.mm(B, mask)-self.C*ones-self.epsilon+z1*z1)*z1)
            z2 = z2 - self.lr * (2*y2*z2 + 2*self.Rho*(-torch.mm(B, mask)+self.C*ones-self.epsilon+z2*z2)*z2)
        return z1, z2

    def update_B(self, B, y1, y2, z1, z2, L, train_mask):
        I = torch.eye(B.shape[0]).to(self.device)
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([B.shape[0], 1]).float().to(self.device)
        for i in range(50):
            B = B - self.lr * (2*(B - I) + 2*self.lambda1*torch.mm(L, B) + torch.mm(y1, mask.T) + self.Rho * torch.mm(torch.mm(B, mask) - self.C * ones - self.epsilon + z1 * z1, mask.T) \
                - torch.mm(y2, mask.T) - self.Rho * torch.mm(-torch.mm(B, mask) + self.C * ones - self.epsilon + z2 * z2, mask.T))
        return B

    def update_y(self, B, y1, y2, z1, z2, train_mask):
        mask = train_mask.reshape([-1, 1]).float()
        ones = torch.ones([y1.shape[0], 1]).float().to(self.device)
        y1 = y1 + self.Rho * (torch.mm(B, mask) - self.C * ones - self.epsilon)
        y2 = y2 + self.Rho * (-torch.mm(B, mask) + self.C * ones - self.epsilon)
        y1 = y1.clamp(min=0)
        y2 = y2.clamp(min=0)
        return y1, y2

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        L = L.to(self.device)
        B = torch.eye(data.x.shape[0]).to(self.device)
        y1 = torch.zeros([data.x.shape[0], 1]).to(self.device)
        y2 = torch.zeros([data.x.shape[0], 1]).to(self.device)
        train_mask = train_mask.to(self.device)
        z1 = torch.ones_like(y1).to(self.device)
        z2 = torch.ones_like(y2).to(self.device)
        for i in range(20):
            z1, z2 = self.update_z(z1, y1, z2, y2, B, train_mask)
            B = self.update_B(B, y1, y2, z1, z2, L, train_mask)
            y1, y2 = self.update_y(B, y1, y2, z1, z2, train_mask)
        return B, data, split_idx


class PPR(object):
    def __init__(self, alpha, args):
        super(PPR, self).__init__()
        self.alpha = alpha
        self.args = args
    
    def process(self, split=0):
        dataset, data, split_idx = get_dataset(self.args, split)
        data.num_classes = data.y.max().item() + 1
        x, adj_t = data.x, data.adj_t
        train_idx = split_idx['train']
        n = data.x.shape[0]
        A = adj_t.to_dense()
        I = torch.diag(torch.ones(n)).to(A)
        A= A + I
        D = torch.sum(A, dim=1) ** (-0.5)
        D[D==float('inf')] = 0
        D = torch.diag(D)
        A = torch.mm(D, A)
        A = torch.mm(A, D)  
        P = torch.ones(n, n).to(A)
        P = P / n
        alpha = self.alpha
        for i in range(self.args.K):
            P = (1-alpha) * torch.matmul(A, P) + alpha * I
        # P = alpha * torch.inverse(I - (1-alpha) * A)
        import ipdb; ipdb.set_trace()
        return P, data, split_idx
    
class PPR_sparse(object):
    def __init__(self, alpha, args):
        super(PPR_sparse, self).__init__()
        self.alpha = alpha
        self.args = args
    
    def process(self, split=0):
        args = self.args
        device = args.device
        device = 'cpu'
        # import ipdb; ipdb.set_trace()
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int().cpu()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        adj = gcn_norm(data.adj_t, add_self_loops=True)
        row, col, value = adj.coo()
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, (data.num_nodes, data.num_nodes)).to(device)
        P = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), torch.ones(data.num_nodes), (data.num_nodes, data.num_nodes)).to(device)
        I = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), torch.ones(data.num_nodes), (data.num_nodes, data.num_nodes)).to(device)
        alpha = self.alpha
        import ipdb; ipdb.set_trace()
        for i in range(self.args.K):
            P = (1-alpha) * torch.matmul(adj, P) + alpha * I
        # P = alpha * torch.inverse(I - (1-alpha) * A)
        return P, data, split_idx


class Pdata(object):
    def __init__(self, lambda1, lr, args):
        super(Pdata, self).__init__()
        self.lambda1 = lambda1
        self.args = args
        self.lr = lr

    def get_data(self, split=0):
        args = self.args
        dataset, data, split_idx = get_dataset(args, split)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        # import ipdb; ipdb.set_trace()
        # adj_t = add_remaining_self_loops(data.adj_t)[0]
        adj = data.adj_t.to_dense()
        adj = adj + torch.eye(adj.shape[0])
        D = torch.sum(adj, dim=1) ** (-0.5)
        D[D==float('inf')] = 0
        D = torch.diag(D)
        adj = torch.mm(torch.mm(D, adj), D)
        L = torch.eye(adj.shape[0]) - adj
        return data, L, train_mask, split_idx

    def process(self, split=0):
        data, L, train_mask, split_idx = self.get_data(split)
        B = torch.eye(data.x.shape[0])
        I = torch.eye(B.shape[0])
        for i in range(20):
            B = B - self.lr * (2*(B - I) + 2*self.lambda1*torch.mm(L, B))
        # A = torch.eye(L.shape[0]) - L
        # B = torch.inverse(I + self.lambda1 * L)
        # alpha = 0.1
        # B = torch.inverse(I - (1-alpha) * A)
        return B, data, split_idx




if __name__ == '__main__':
    args = parse_args()
    lambda1 = 0.2 # alpha 0.8
    blr = 0.01
    C = 0.2
    Rho = 0.01
    blr = 0.01
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    sp = Fdata_sparse1(lambda1, C, Rho, blr, device, args)
    sp.process()
