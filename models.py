import pdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP, GATConv, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layers import GCN, BernPool


class GraphClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(GraphClassificationModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

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

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, kl_loss1, ref_loss1 = self.bern_pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, kl_loss2, ref_loss2 = self.bern_pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, kl_loss3, ref_loss3 = self.bern_pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x, kl_loss1 + kl_loss2 + kl_loss3, ref_loss1 + ref_loss2 + ref_loss3