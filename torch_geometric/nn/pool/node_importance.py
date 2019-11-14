import torch
from torch_geometric.nn import GraphConv, GATConv, SAGEConv
from torch_geometric.nn.pool.topk_pool import topk
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import uniform_


class NodeImportance(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, layer=1, gnn='GCN', bias=True, **kwargs):
        super(NodeImportance, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.layer = layer

        assert gnn in ['GCN', 'GAT', 'SAGE']
        if gnn == 'GCN':
            if layer == 1:
                self.gnn = GraphConv(self.in_channels, 1, **kwargs)
            elif layer == 2:
                self.gnn1 = GraphConv(self.in_channels, self.in_channels, **kwargs)
                self.gnn2 = GraphConv(self.in_channels, 1, **kwargs)
            elif layer == 3:
                self.gnn1 = GraphConv(self.in_channels, self.in_channels, **kwargs)
                self.gnn2 = GraphConv(self.in_channels, self.in_channels, **kwargs)
                self.gnn3 = GraphConv(self.in_channels, 1, **kwargs)
        elif gnn == 'GAT':
            self.gnn = GATConv(self.in_channels, 1, **kwargs)
        else:
            self.gnn = SAGEConv(self.in_channels, 1, **kwargs)

        self.weight_closeness = Parameter(torch.Tensor(1))
        self.weight_degree = Parameter(torch.Tensor(1))
        self.weight_score = Parameter(torch.Tensor(1))

        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.layer == 1:
            self.gnn.reset_parameters()
        elif self.layer == 2:
            self.gnn1.reset_parameters()
            self.gnn2.reset_parameters()
        elif self.layer == 3:
            self.gnn1.reset_parameters()
            self.gnn2.reset_parameters()
            self.gnn3.reset_parameters()

        uniform_(self.weight_closeness, a=0, b=1)
        uniform_(self.weight_degree, a=0, b=1)
        uniform_(self.bias, a=0, b=1)
        uniform_(self.weight_score, a=0, b=1)


    def forward(self, x, edge_index, closeness, degree, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if self.layer == 1:
            score = torch.relu(self.gnn(x, edge_index).view(-1))
        elif self.layer == 2:
            score = torch.relu(self.gnn1(x, edge_index))
            score = torch.relu(self.gnn2(score, edge_index).view(-1))
        elif self.layer == 3:
            score = torch.relu(self.gnn1(x, edge_index))
            score = torch.relu(self.gnn2(score, edge_index))
            score = torch.relu(self.gnn3(score, edge_index).view(-1))

        '''centrality adjust'''
        closeness = closeness * self.weight_closeness
        degree = degree * self.weight_degree
        centrality = closeness + degree
        if self.bias is not None:
            centrality += self.bias

        score = score * self.weight_score
        score = score + centrality
        score = F.relu(score)

        perm = topk(score, self.ratio, batch)
        tmp1 = x[perm]
        tmp2 = score[perm]
        x = tmp1 * tmp2.view(-1, 1)
        batch = batch[perm]

        return x, perm, batch

    def __repr__(self):
        return '{}({}, {}, ratio={})'.format(self.__class__.__name__,
                                             self.gnn_name, self.in_channels,
                                             self.ratio)
