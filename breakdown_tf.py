import numpy as np
import math
import itertools
import scipy as sp
import random
import tensorflow as tf
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, to_undirected, remove_self_loops
from torch_scatter import scatter_add
import networkx as nx
'''

dataset = Planetoid(root='data/', name='cora', split='public')
data = dataset.data
#Cora  = CoraDataset()
A = np.zeros((data.num_nodes, data.num_nodes))
A[data.edge_index[0], data.edge_index[1]] = 1
edge_index = data.edge_index
#edge_index = remove_self_loops(edge_index)[0]
num_nodes = data.num_nodes

step_size = 1.
input_dim = data.x.shape[-1]
hidden_dim = 32
output_dim = 7
#linear = nn.Linear(input_dim, output_dim)
linear = tf.keras.models.Sequential()
linear.add(tf.keras.Input(shape=(input_dim,)))
linear.add(tf.keras.layers.Dense(output_dim, activation=None))
#sheaf_learner = nn.Linear(2*input_dim, 1, bias=False)
initializer = tf.keras.initializers.Ones()
sheaf_learner = tf.keras.models.Sequential()
sheaf_learner.add(tf.keras.Input(shape=(2*input_dim,)))
sheaf_learner.add(tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer))

# index each edge
edge_to_idx = dict()
for e in range(edge_index.shape[1]):
    source = edge_index[0, e].item()
    target = edge_index[1, e].item()
    edge_to_idx[(source, target)] = e

# index of each edge
left_index, right_index = [], []
for e in range(edge_index.shape[1]):
    source = edge_index[0, e].item()
    target = edge_index[1, e].item()
    left_index.append(e)
    right_index.append(edge_to_idx[(target, source)])

#left_index = torch.tensor(left_index, dtype=torch.long, device=edge_index.device)
#right_index = torch.tensor(right_index, dtype=torch.long, device=edge_index.device)
left_right_index = tf.stack([left_index, right_index])

x = data.x
# define perceptron on 
row, col = edge_index
x_row = tf.gather(x, row)
x_col = tf.gather(x, col)
maps = sheaf_learner(tf.concat([x_row, x_col], axis=1))
maps = tf.tanh(maps)

#left_maps = torch.index_select(maps, index=left_index, dim=0)
left_maps = tf.gather(maps, left_index)
#right_maps = torch.index_select(maps, index=right_index, dim=0)
right_maps = tf.gather(maps, right_index)
non_diag_maps = -left_maps * left_maps
# reduce to size N (size of the graph)
ref = tf.Variable(tf.zeros(num_nodes, dtype = tf.float32))
indices = tf.constant(row.reshape(-1,1))
updates = maps**2
diag_maps = tf.compat.v1.scatter_add(ref, indices, updates)

d_sqrt_inv = tf.math.pow(diag_maps + 1, -0.5)
left_norm, right_norm = tf.gather(d_sqrt_inv,row.numpy()), tf.gather(d_sqrt_inv,col.numpy())
norm_maps = tf.reshape(left_norm, (-1,1)) * non_diag_maps * tf.reshape(right_norm, (-1,1))
diag = d_sqrt_inv * diag_maps * d_sqrt_inv
diag = tf.reshape(diag, (-1,1))

diag_indices = tf.stack((tf.range(num_nodes), tf.range(num_nodes)))
all_indices = tf.concat([diag_indices, edge_index], axis=1)
all_values = tf.concat([tf.reshape(diag, -1), tf.reshape(norm_maps, -1)], axis=0)
laplacian = tf.sparse.SparseTensor(tf.cast(tf.transpose(all_indices), dtype=tf.int64), all_values, dense_shape=(num_nodes, num_nodes))
laplacian = tf.sparse.reorder(laplacian)
laplacian = tf.sparse.to_dense(laplacian)
