"""Scaled Smoothing Splines utilities."""
import numpy as np
import torch
from torch.nn import ReLU
from torch_scatter import scatter_add
from torch_geometric.utils import to_networkx
from networkx.algorithms.centrality import closeness_centrality, degree_centrality


def get_centrality(x, edge_index, batch):
    num_graphs = batch[-1] + 1
    N = x.shape[0]
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    cum_num_nodes = torch.cat((cum_num_nodes, torch.tensor([N]).cuda()))
    row, col = edge_index
    c_centrality = []
    d_centrality = []
    for i in range(num_graphs):
        '''each graph'''
        s_id = cum_num_nodes[i]
        e_id = cum_num_nodes[i + 1]
        mask = torch.eq(row, s_id)
        for node in range(s_id + 1, e_id):
            mask = mask + torch.eq(row, node)
        g_row = torch.masked_select(row, mask) - s_id
        g_col = torch.masked_select(col, mask) - s_id

        G = to_networkx(torch.stack([g_row, g_col], dim=0))
        c_centrality = c_centrality + list(closeness_centrality(G).values())
        d_centrality = d_centrality + list(degree_centrality(G).values())

    c_centrality = torch.Tensor(c_centrality).cuda()
    d_centrality = torch.Tensor(d_centrality).cuda()
    return c_centrality, d_centrality


def get_closeness_centrality(dataset):
    centrality = []
    for data in dataset:
        '''each graph'''
        g_row, g_col = data.edge_index
        G = to_networkx(torch.stack([g_row, g_col], dim=0))
        c_centrality = list(closeness_centrality(G).values())
        centrality = centrality + c_centrality

    return centrality


def get_degree_centrality(dataset):
    centrality = []
    for data in dataset:
        '''each graph'''
        g_row, g_col = data.edge_index
        G = to_networkx(torch.stack([g_row, g_col], dim=0))
        c_centrality = list(degree_centrality(G).values())
        centrality = centrality + c_centrality

    return centrality


def save_file(name, file, property):
    path = '../data/' + name + '/' + property + '.npy'
    np.save(path, file)
    print('save ' + name + '\'s ' + property + ' successfully!')


def triple_point_multi(x):
    return x * x * x


def d_k(k, x, basis_x, M, epsilon):
    m = ReLU()
    a = m(triple_point_multi(x - basis_x[k]))
    b = m(triple_point_multi(x - basis_x[M]))
    numerator = a - b
    denominator = basis_x[M] - basis_x[k] + epsilon
    res = torch.zeros(numerator.shape[-1]).cuda()
    res = torch.addcdiv(res, value=1, tensor1=numerator, tensor2=denominator, out=None)
    return res


def normalize_perm(topk_x, perm, batch, M):
    '''norm_perm'''
    num_graphs = batch[-1] + 1
    N = topk_x.shape[0]

    size1 = num_graphs * M
    size2 = topk_x.shape[1]
    norm_perm = perm.new_zeros(size1)
    norm_batch = batch.new_zeros(size1)
    norm_x = topk_x.new_zeros((size1, size2))

    num_nodes = scatter_add(batch.new_ones(topk_x.size(0)), batch, dim=0)
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    cum_num_nodes = torch.cat((cum_num_nodes, torch.tensor([N]).cuda()))

    for i in range(num_graphs):
        '''select x'''
        s_id = cum_num_nodes[i]
        e_id = cum_num_nodes[i + 1]
        x = topk_x[s_id:e_id]

        '''select perm'''
        mask = batch.eq(i)
        num_nodes = mask.sum().item()
        indices = torch.masked_select(perm, mask)

        l_id = i * M
        '''norm batch'''
        norm_batch[l_id:l_id + M] = batch.new([i for j in range(M)])

        if num_nodes >= M:
            norm_x[l_id:l_id + M] = x[0:M]
            norm_perm[l_id:l_id + M] = indices[0:M]
        else:
            mid_id = l_id + num_nodes
            '''norm x'''
            norm_x[l_id:mid_id] = x
            extra_x = x.new_zeros(M - num_nodes, x.shape[1])
            norm_x[mid_id:l_id + M] = extra_x

            '''norm perm'''
            norm_perm[l_id:mid_id] = indices
            extra_id = indices.new_zeros(M - num_nodes)
            norm_perm[mid_id:l_id + M] = extra_id

    # norm_perm = torch.zeros((num_graphs, M)).long().cuda()
    # for i in range(num_graphs):
    #     if len(perm[i]) >= M:
    #         norm_perm[i] = perm[i][:M]
    #     else:
    #         extra_id = torch.LongTensor([0 for j in range(M - len(perm[i]))]).cuda()
    #         basis_id = torch.cat((perm[i][:len(perm[i])], extra_id), dim=0)
    #         norm_perm[i] = basis_id
    # '''sort norm_perm ascending'''
    # idx = torch.LongTensor([i for i in range(M - 1, -1, -1)]).cuda()  # create inverted indices
    # norm_perm = norm_perm.index_select(1, idx)
    return norm_x, norm_perm, norm_batch


def smooth_spline_learned(x, batch, topk_x, M, sort_feature, epsilon):
    N = x.shape[0]
    num_graphs = batch[-1] + 1
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    cum_num_nodes = torch.cat((cum_num_nodes, torch.tensor([N]).cuda()))

    '''compute N_1, N_2'''
    N_1 = x.new_ones((N, 1))
    N_2 = x
    expand_x = torch.cat((N_1, N_2), dim=-1)
    '''compute N_k+2, k=1,...,M'''
    N_k = torch.zeros((N, M * x.shape[-1])).cuda()
    idx = torch.LongTensor([i for i in range(M - 1, -1, -1)]).cuda()  # create inverted indices
    for i in range(num_graphs):  # for each graph
        s_id = cum_num_nodes[i]
        e_id = cum_num_nodes[i + 1]
        # knot = x[perm[i * M:i * M + M]]
        knot = topk_x[i * M:i * M + M]
        basis_x = x[s_id:e_id]

        # '''sort basis'''
        if sort_feature:
            basis_node, _ = knot.sort(dim=0, descending=False)

        '''sort norm_perm ascending'''
        knot = knot.index_select(0, idx)

        l_id = 0
        for j in range(0, M):
            N = d_k(j, basis_x, knot, M - 1, epsilon) - d_k(M - 1, basis_x, knot, M - 1, epsilon)
            r_id = l_id + x.shape[-1]
            N_k[s_id:e_id, l_id:r_id] = N
            l_id = r_id

    expand_x = torch.cat((expand_x, N_k), dim=-1)

    return expand_x
