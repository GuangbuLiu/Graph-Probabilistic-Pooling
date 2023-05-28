import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import degree, to_dense_batch, to_dense_adj, to_networkx, to_undirected, subgraph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import filter_adj
from torch_scatter import scatter_add
import math
import matplotlib.pyplot as plt

class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)



class BernPool(torch.nn.Module):
    def __init__(self,in_channels,args, num_layer):
        super(BernPool, self).__init__()
        self.in_channels=in_channels
        self.ratio=args.pool_ratio

        self.ref = Parameter(torch.Tensor(args.n_refpoint, self.in_channels),requires_grad=True)
        nn.init.orthogonal(self.ref.data)
        c = np.array([1])
        self.c = nn.Parameter(torch.Tensor(c), requires_grad = True)
        self.lin1 = torch.nn.Linear(args.n_refpoint,1)

        self.weight = nn.Parameter(torch.FloatTensor(self.in_channels, self.in_channels)).cuda()
        nn.init.xavier_normal_(self.weight)


    def cosine_similarity(self,x,y):

        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
        cos_dis = torch.mm(x, torch.transpose(y, 0, 1))

        return cos_dis

    def mask_to_index(self,mask):
        '''Convert a mask to an index representation '''

        return mask.nonzero(as_tuple=False).view(-1)

    def find_del_graph(self,batch,pool_batch):
        index_batch = torch.unique(batch)
        index_pool_batch = torch.unique(pool_batch)

        indices = torch.ones_like(index_batch).cuda()
        for i in index_pool_batch:
            indices = indices & (index_batch != i)

        del_graph_index = self.mask_to_index(indices)
        return del_graph_index

    def KL_divgence(self, prob):

        expect_prob = torch.ones_like(prob) * self.ratio
        kl_div = prob * torch.log(prob / expect_prob) + (1 - prob) * torch.log((1 - prob) / (1 - expect_prob))
        kl_loss = torch.sum(kl_div)
        return kl_loss

    def orthogonal_loss(self):

        I = torch.eye(self.ref.shape[0]).cuda()
        ortho_regularization = torch.matmul(self.ref, torch.transpose(self.ref, 0, 1)) - self.c * I
        ref_loss = torch.norm(ortho_regularization, p = "fro")
        return ref_loss

    def filter_feature(self, x, edge_index, edge_attr, batch, bern_mask):

        if len(torch.unique(batch)) != len(torch.unique(batch[self.mask_to_index(bern_mask)])):
            del_graph_index = self.find_del_graph(batch, batch[self.mask_to_index(bern_mask)])
            for index in del_graph_index:
                node_index = self.mask_to_index(batch == index)
                bern_mask[node_index] = 1

        mask_index = self.mask_to_index(bern_mask)
        pool_batch = batch[mask_index]

        # Bernoulli Node Clustering
        edge_index, norm = GCN.norm(edge_index, x.size(0), edge_weight=None, dtype=x.dtype)
        A = to_dense_adj(edge_index, max_num_nodes=x.size(0), edge_attr=norm,
                         batch=edge_index.new_zeros(x.size(0))).squeeze(0)
        sample_matrix_adj = A[mask_index]
        pool_x_clustering = torch.matmul(sample_matrix_adj, x)

        # Bernoulli Node Dropping
        sample_matrix_diag = torch.diag(bern_mask)
        pool_x_dropping = torch.mm(sample_matrix_diag[mask_index], x)

        # Hybrid learning
        pool_x = pool_x_dropping + pool_x_clustering
        pool_x = F.relu(torch.matmul(pool_x, self.weight))

        pool_edge_index, pool_edge_attr = filter_adj(edge_index, edge_attr, mask_index, num_nodes=x.size(0))

        return pool_x, pool_edge_index, pool_edge_attr, pool_batch, mask_index


    def forward(self,x, edge_index, edge_attr, batch):

        similarity = self.cosine_similarity(x, self.ref)
        score = self.lin1(similarity).squeeze()
        score = torch.sigmoid(score)
        bern_mask = torch.bernoulli(score)

        pool_x, pool_edge_index, pool_edge_attr, pool_batch, mask_index = self.filter_feature(x, edge_index, edge_attr, batch, bern_mask)
        kl_loss = self.KL_divgence(score.mean())
        ref_loss= self.orthogonal_loss()

        return pool_x, pool_edge_index, pool_edge_attr, pool_batch, kl_loss, ref_loss

