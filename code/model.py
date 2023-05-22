import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn.models import LabelPropagation
from prop import Propagation
import math


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, args, prop, **kwargs):
        super(MLP, self).__init__()
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
        # self.prop = prop
        self.args = args

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, **kwargs):
        x = data.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, **kwargs):
        super(GCN, self).__init__()
        # num_layers = args.num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, normalize=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True, normalize=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, args=None, **kwargs):
        x, adj_t, = data.x, data.adj_t
        # import ipdb; ipdb.set_trace()
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x
        return x.log_softmax(dim=-1)



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, **kwargs):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, **kwargs):
        x, adj_t, = data.x, data.adj_t
        x = F.relu(self.conv1(x, adj_t))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, heads=1, output_heads=1, **kwargs):
        super(GAT, self).__init__()
        conv_layer = GATConv
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        heads = 4
        hidden_channels1 = 50
        self.convs.append(conv_layer(in_channels, hidden_channels1, heads=heads))
        self.dropout = dropout
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels1, heads=heads))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(conv_layer(hidden_channels, out_channels, heads=output_heads, concat=False))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, data, **kwargs):
        x, adj_t, = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            # x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)




class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(SGC, self).__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=3, cached=True)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data, **kwargs):
        x, adj_t = data.x, data.adj_t
        # x, adj_t = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, adj_t)
        return F.log_softmax(x, dim=1)

class APPNP1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super(APPNP1, self).__init__()
        num_layers = args.num_layers
        self.hidden_channels = hidden_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.prop = prop
        self.args = args

    def propagate(self, data):
        x, adj_t, = data.x, data.adj_t
        self.x = self.prop(x, adj_t, data=data, alpha=self.args.pro_alpha)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        # x = self.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if data.f is None:
            diff = torch.zeros_like(x)
            loss1 = torch.sum(diff * diff)
        else:
            label = data.f
            train_mask = data.train_mask
            total_weight = torch.zeros(x.shape[0]).cuda()
            total_weight[train_mask] = 1
            num_class = data.y.max() + 1
            for i in range(num_class):
                weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(num_class)
                _, index = label.max(dim=1)
                pos = (index == i)
                # test_distribution.append(sum(pos).item())
                weight[~pos] = 0
                weight[train_mask] = 0
                value, indices = torch.topk(weight, 70)
                total_weight[indices] = value
            diff = data.f - F.softmax(x, dim=-1)
            diff = torch.sum(diff * diff, 1)
            loss1 = torch.sum(total_weight * diff)
        x = self.prop(x, adj_t, data=data)
        data.f = torch.clone(x).detach()
        data.f = F.softmax(data.f, dim=-1)
        return F.log_softmax(x, dim=1), loss1


class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super(APPNP, self).__init__()
        num_layers = args.num_layers
        self.hidden_channels = hidden_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.prop = prop
        self.args = args

    def propagate(self, data):
        x, adj_t, = data.x, data.adj_t
        self.x = self.prop(x, adj_t, data=data, alpha=self.args.pro_alpha)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data, args=None):
        x, adj_t, = data.x, data.adj_t
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # import ipdb; ipdb.set_trace()
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = self.prop(x, adj_t, data=data)
        return F.log_softmax(x, dim=1)


class IAPPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super(IAPPNP, self).__init__()
        num_layers = args.num_layers
        self.hidden_channels = hidden_channels
        self.lins = torch.nn.ModuleList()
        # in_channels = 256
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.prop = prop

    def propagate(self, data):
        x, adj_t, = data.x, data.adj_t
        self.x = self.prop(x, adj_t, data=data)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x = self.x
        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        # return x
        return F.log_softmax(x, dim=1)


## prop in the middle layer
class APPNP_Hidden(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, **kwargs):
        super(APPNP_Hidden, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        # self.lin2 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels*2, out_channels)
        self.dropout = dropout
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        # print(f'input: x: {x}')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        # print(f'lin1 weight: {self.lin1.weight}')
        # import ipdb; ipdb.set_trace()
        # print(f'lin1: x: {x}')
        # x.requires_grad_(); x.register_hook(lambda grad: print(f'backward before prop x: {grad}'))
        x = self.prop(x, adj_t, data=data)
        # x.requires_grad_(); x.register_hook(lambda grad: print(f'backward after prop x: {grad}, min: {grad.min()}, max: {grad.max()}'))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # x.requires_grad_(); x.register_hook(lambda grad: print(f'backward output x: {grad}, min: {grad.min()}, max: {grad.max()}'))
        # print(f'lin2: x: {x}')
        return F.log_softmax(x, dim=1)

## TODO: model: concatnate node and graph features

class APPNP_Concat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, dataset, **kwargs):
        super(APPNP_Concat, self).__init__()

        from util_mf import gcn_norm_ours, spectral_embedding, Eigendecomposition_loss
        edge_index = gcn_norm_ours(edge_index=dataset[0].adj_t, edge_weight=None, num_nodes=dataset[0].x.size(0), add_self_loops=True, dtype=None)
        dim = 64
        # import ipdb; ipdb.set_trace()
        self._cached_embedding = Eigendecomposition_loss(edge_index.cuda(), dim=dim, device='cuda')

        # hidden_channels = hidden_channels * 2
        self.lin1 = Linear(in_channels, hidden_channels)
        # self.lin1 = Linear(in_channels+dim, hidden_channels)
        # self.lin1 = Linear(dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.prop = prop


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        # import ipdb; ipdb.set_trace()
        # x = torch.cat([x, self._cached_embedding], dim=1)
        # x = self._cached_embedding
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t, data=data)
        return F.log_softmax(x, dim=1)

class CSMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, args, **kwargs):
        super(CSMLP, self).__init__()
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
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, **kwargs):
        x = data.x
        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class CorrectAndSmooth(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        super(CorrectAndSmooth, self).__init__()
        num_correct_layer = args.num_correct_layer
        num_smooth_layer = args.num_smooth_layer
        correct_alpha = args.correct_alpha
        smooth_alpha = args.smooth_alpha
        self.prop1 = Propagation(K=num_correct_layer, alpha=correct_alpha, mode='CS', cached=True, args=args)
        self.prop2 = Propagation(K=num_smooth_layer, alpha=smooth_alpha, mode='CS', cached=True, args=args)

    def init_label(self, data):
        mask = data.train_mask
        nodes = data.x.shape[0]
        classes = data.y.max() + 1
        label = torch.zeros(nodes, classes).cuda()
        label[mask, data.y[mask]] = 1
        return label

    def correct(self, data, mlp, edge_weight=None, alpha=None):
        x, adj_t = data.x, data.adj_t
        label = self.init_label(data)
        mask = data.train_mask
        error = torch.zeros_like(mlp)
        error[mask] = label[mask] - mlp[mask]
        smoothed_error = self.prop1(error, adj_t, post_step=lambda z: z.clamp_(-1., 1.), edge_weight=edge_weight, alpha=alpha)
        sigma = error[mask].abs().sum() / int(mask.sum())
        scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
        scale[scale.isinf() | (scale > 1000)] = 1.0
        return mlp + scale * smoothed_error

    def smooth(self, data, y_soft, edge_weight=None, alpha=None):
        # print(y_soft[0])
        x, adj_t = data.x, data.adj_t
        mask = data.train_mask
        label = self.init_label(data)
        y_soft[mask] = label[mask]
        r = self.prop2(y_soft, adj_t, post_step=lambda x: x.clamp(0, 1),edge_weight=edge_weight, alpha=alpha)
        # r = self.prop2(y_soft, adj_t, post_step=lambda x: x, edge_weight=edge_weight)
        return r



class ORTGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, args, **kwargs):
        super(ORTGNN, self).__init__()
        num_layers = args.num_layers
        self.hidden_channels = hidden_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.prop = Propagation(K=args.K, alpha=args.alpha, mode='ORTGNN', cached=True, args=args)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()

    def propagate(self, data):
        x, adj_t, = data.x, data.adj_t
        self.x = self.prop(x, adj_t, data=data)
        return x

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        # x = self.x
        # print(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = self.prop(x, adj_t, data=data)
        if torch.isnan(x).any():
            import ipdb
            ipdb.set_trace()
        return F.log_softmax(x, dim=1)


class LP(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        super(LP, self).__init__()
        self.prop = Propagation(K=args.K, alpha=args.alpha, mode='CS', cached=True, args=args)

    def init_label(self, data):
        mask = data.train_mask
        nodes = data.x.shape[0]
        classes = data.y.max() + 1
        label = torch.zeros(nodes, classes).cuda()
        label[mask, data.y[mask]] = 1
        return label

    def forward(self, data):
        label = self.init_label(data)
        x, adj_t = data.x, data.adj_t
        plabel = self.prop(label, adj_t, post_step=lambda x:torch.clamp(x,0,1))
        # plabel = self.prop(label, adj_t)
        return plabel


class SIGN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, args, **kwargs):
        super(SIGN, self).__init__()
        num_layers = args.num_layers
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = torch.nn.Linear((num_layers + 1) * hidden_channels,
                                   out_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, data):
        # import ipdb; ipdb.set_trace()
        # xs = data.xs
        outs = None
        for x, bn, lin in zip(data.xs, self.bns, self.lins):
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            # x = bn(x)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
            # outs.append(out)
            if outs is None:
                outs = x
            else:
                outs = torch.cat((outs, x), dim=-1)
            # torch.cuda.empty_cache()
        # outs = torch.cat(outs, dim=-1)
        x = self.lin(outs)
        # x = self.bn(x)
        # x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        # x = self.lin1(x)
        return torch.log_softmax(x, dim=-1)



