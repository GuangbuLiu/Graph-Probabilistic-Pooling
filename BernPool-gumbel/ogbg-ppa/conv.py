import pdb
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import degree, to_dense_batch, to_dense_adj, to_networkx, to_undirected, subgraph
from torch.nn import Parameter
from torch_geometric.nn.pool.topk_pool import filter_adj
from torch_scatter import scatter_add
import math

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out



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

        # return self.propagate(edge_index, x=x, norm=norm)
        return self.propagate(edge_index, x=x, edge_attr=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)



class BernPool(torch.nn.Module):
    def __init__(self,in_channels, pool_ratio = 0.8, refpoint = 32):
        super(BernPool, self).__init__()
        self.in_channels=in_channels
        self.ratio = pool_ratio

        self.ref = Parameter(torch.Tensor(refpoint, self.in_channels),requires_grad=True)
        nn.init.orthogonal(self.ref.data)
        c = np.array([1])
        self.c = nn.Parameter(torch.Tensor(c), requires_grad = True)
        self.lin1 = torch.nn.Linear(refpoint,1)

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
        # pdb.set_trace()
        similarity = self.cosine_similarity(x, self.ref)
        score = self.lin1(similarity).squeeze()
        score = torch.sigmoid(score)
        # pdb.set_trace()
        bern_mask = torch.bernoulli(score)

        pool_x, pool_edge_index, pool_edge_attr, pool_batch, mask_index = self.filter_feature(x,
                                                                    edge_index, edge_attr, batch, bern_mask)
        kl_loss = self.KL_divgence(score.mean())
        ref_loss= self.orthogonal_loss()

        return pool_x, pool_edge_index, pool_edge_attr, pool_batch, kl_loss, ref_loss


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "sum", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.pool = BernPool(emb_dim, pool_ratio=0.8)
        # self.pool = SAGPooling(in_channels=emb_dim, ratio=0.8)

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        x_list = []
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # SAGPool
                # h, edge_index, edge_attr, batch, _, _ = self.pool(h, edge_index, edge_attr, batch)

                # BernPool
                h, edge_index, edge_attr, batch, kl_loss, ref_loss = self.pool(h, edge_index, edge_attr, batch)

                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            x_list.append(torch.cat([gmp(h, batch), gap(h, batch)], dim=1))
            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        node_representation = 0
        for layer in range(self.num_layer):
            node_representation += x_list[layer]

        return node_representation, kl_loss, ref_loss


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
