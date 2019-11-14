import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, NodeImportance
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
from torch_geometric.utils import degree
import numpy as np
from utils.graph import *
import argparse

parser = argparse.ArgumentParser(description='GSSNN.')
parser.add_argument('--dataset', type=str, default='MUTAG', metavar='dataset',
                    help='input batch size for training (default: MUTAG)')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--epsilon', type=float, default=1e-6)
parser.add_argument('--M1', type=int, default=5)
parser.add_argument('--M2', type=int, default=5)
parser.add_argument('--ratio1', type=float, default=0.4)
parser.add_argument('--ratio2', type=float, default=0.4)
parser.add_argument('--seed', type=int, default=1233)
parser.add_argument('--sort_feature', type=bool, default=False)
parser.add_argument('--ss_layer', type=int, default=2)
parser.add_argument('--pool_layer', type=int, default=2)
parser.add_argument('--conv_layer', type=int, default=3)
parser.add_argument('--add_knot', type=bool, default=True)
parser.add_argument('--epoch', type=int, default=101)
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
name = args.dataset
print(name)
print(args)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
dataset = TUDataset(path, name=name).shuffle()


def build_x():
    num_nodes = dataset.slices['x'][-1]
    node_tags = []
    for i in range(len(dataset)):
        node_tags.append(degree(dataset[i].edge_index[0]).tolist())

    # Extracting unique tag labels
    tagset = set([])
    for tag in node_tags:
        tagset = tagset.union(set(tag))
    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    dataset.data.x = torch.zeros(num_nodes, len(tagset))
    dataset.data.num_nodes = dataset.slices['x'][-1]

    for i in range(len(dataset)):
        dataset[i].x[range(len(node_tags[i])), [tag2index[tag] for tag in node_tags[i]]] = 1


if name in ('IMDB-BINARY', 'IMDB-MULTI', 'COLLAB'):
    print('NO input feature.')
    node_slice = [0]
    for data in dataset:
        node_slice.append(data.num_nodes + node_slice[-1])
    dataset.slices['x'] = torch.LongTensor(node_slice)

    build_x()

degree = torch.Tensor(np.load('../data/' + name + '/degree.npy'))
closeness = torch.Tensor(np.load('../data/' + name + '/closeness.npy'))
dataset.data.__setitem__('degree', degree)
dataset.data.__setitem__('closeness', closeness)
dataset.slices['degree'] = dataset.slices['x']
dataset.slices['closeness'] = dataset.slices['x']

test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
batch_size = args.batch_size
test_loader = DataLoader(test_dataset, batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

M1 = args.M1
M2 = args.M2
ratio1 = args.ratio1
ratio2 = args.ratio2
dim = args.dim
epsilon = args.epsilon


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features

        expand_feat1 = 1 + (M1 + 1) * num_features
        expand_feat2 = 1 + (M2 + 1) * dim
        self.smooth_fc1 = Linear(expand_feat1, num_features)
        if args.ss_layer > 1:
            self.smooth_fc2 = Linear(expand_feat2, dim)

        self.conv1 = GCNConv(num_features, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(args.conv_layer - 1):
            self.convs.append(GCNConv(dim, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

        if args.add_knot:
            if args.ss_layer == 1:
                self.fc1 = Linear(num_features + dim, dim)
            elif args.ss_layer == 2:
                self.fc1 = Linear(num_features + dim + dim, dim)
        else:
            self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

        self.pool1 = NodeImportance(num_features, ratio1, layer=args.pool_layer)

    def reset_parameters(self):
        self.smooth_fc1.reset_parameters()
        if args.ss_layer > 1:
            self.smooth_fc1.reset_parameters()

        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.pool1.reset_parameters()

    def forward(self, x, edge_index, batch, y, degree, closeness):
        '''ss 1'''
        if args.ss_layer >= 1:
            topk_x, perm, topk_batch = self.pool1(x, edge_index, closeness, degree, batch=batch)
            topk_x, perm, topk_batch = normalize_perm(topk_x, perm, topk_batch, M1)
            x = smooth_spline_learned(x, batch, topk_x, M1, args.sort_feature, args.epsilon)
            x = self.smooth_fc1(x)
            x1 = gmp(x[perm], topk_batch)

        '''conv1'''
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        '''ss 2'''
        if args.ss_layer >= 2:
            x = smooth_spline_learned(x, batch, x[perm], M2, args.sort_feature, args.epsilon)
            x = self.smooth_fc2(x)
            x2 = gmp(x[perm], topk_batch)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, edge_index))
            x = bn(x)

        '''global_pool'''
        if args.ss_layer == 1:
            basis_node = x1
        elif args.ss_layer == 2:
            basis_node = torch.cat((x1, x2), dim=-1)
        x = gap(x, batch)
        if args.add_knot:
            x = torch.cat((x, basis_node), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.8 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch, data.y, data.degree, data.closeness)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch, data.y, data.degree, data.closeness)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


high_acc = 0
for epoch in range(1, args.epoch):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.3f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
    if test_acc >= high_acc:
        high_acc = test_acc
        high_train_acc = train_acc

log = 'Train: {:.4f}, Test: {:.4f}'
print(log.format(high_train_acc, high_acc))
print(args)
