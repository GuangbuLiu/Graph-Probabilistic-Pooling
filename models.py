import pdb
from math import ceil
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP, GATConv, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# from layers import GNN
from torch_geometric.nn import dense_diff_pool
from layers import GCN, MVPool,BernPool


class GraphClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(GraphClassificationModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        # self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        # c = np.array([1])
        # self.c = nn.Parameter(torch.Tensor(c), requires_grad = True)
        # self.dic = nn.Parameter(torch.Tensor(args.n_dic, self.nhid), requires_grad=True)
        # nn.init.orthogonal(self.dic.data)

        # self.mlp1 = torch.nn.Linear(self.nhid, self.nhid)
        # self.mlp2 = torch.nn.Linear(self.nhid, self.nhid)
        # self.mlp3 = torch.nn.Linear(self.nhid, self.nhid)

        self.bern_pool1 = BernPool(self.nhid, args, num_layer= 1)
        self.bern_pool2 = BernPool(self.nhid, args, num_layer= 2)
        self.bern_pool3 = BernPool(self.nhid, args, num_layer= 3)


        # GCN
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        # GAT
        # self.conv1 = GATConv(self.num_features, self.nhid)
        # self.conv2 = GATConv(self.nhid, self.nhid)
        # self.conv3 = GATConv(self.nhid, self.nhid)

        # GraphSAGE
        # self.conv1 = SAGEConv(self.num_features, self.nhid)
        # self.conv2 = SAGEConv(self.nhid, self.nhid)
        # self.conv3 = SAGEConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data, epoch):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, kl_loss1, dic_loss1 = self.bern_pool1(x, edge_index, edge_attr, batch, epoch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, kl_loss2, dic_loss2 = self.bern_pool2(x, edge_index, edge_attr, batch, epoch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, kl_loss3, dic_loss3 = self.bern_pool3(x, edge_index, edge_attr, batch, epoch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x, kl_loss1 + kl_loss2 + kl_loss3, dic_loss1 + dic_loss2 + dic_loss3



class NodeClassificationModel(torch.nn.Module):
    def __init__(self, args, sum_res=False, act=F.relu):
        super(NodeClassificationModel, self).__init__()
        assert args.depth >= 1
        self.in_channels = args.num_features
        self.hidden_channels = args.nhid
        self.out_channels = args.num_classes
        self.depth = args.depth
        self.pool_ratios = [args.pool1, args.pool2, args.pool3, args.pool4, args.pool5]
        self.act = act
        self.sum_res = sum_res

        channels = self.hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(self.in_channels, channels))
        for i in range(self.depth):
            self.pools.append(MVPool(channels, self.pool_ratios[i], args))
            self.down_convs.append(GCN(channels, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth):
            self.up_convs.append(GCN(in_channels, channels))
        self.up_convs.append(GCN(channels, self.out_channels))

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = F.dropout(x, p=0.92, training=self.training)
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, edge_weight, batch, perm = self.pools[i - 1](x, edge_index, edge_weight, batch)
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x)
        x = self.up_convs[-1](x, edge_index, edge_weight)

        return x
