'''Process Data and Compute Graph Centrality'''
import os.path as osp
from torch_geometric.datasets import TUDataset
from utils.graph import *
import argparse

parser = argparse.ArgumentParser(description='Process Data and Compute Graph Centrality.')
parser.add_argument('--dataset', default='MUTAG', type=str)
args = parser.parse_args()

name = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
dataset = TUDataset(path, name=name).shuffle()

if osp.exists(path + '/degree.npy'):
    print('Already exist centrality information.')
else:
    degree = get_degree_centrality(dataset)
    save_file(name, degree, 'degree')
    closeness = get_closeness_centrality(dataset)
    save_file(name, closeness, 'closeness')

print('Process Done!')
