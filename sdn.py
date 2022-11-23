import numpy as np
import math
import itertools
import scipy as sp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import Planetoid, ZINC, GNNBenchmarkDataset
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_add
from torch_geometric.utils import to_dense_adj, to_undirected, remove_self_loops
from torch.nn import Embedding, Linear
from torch.nn import Parameter
import pdb
#for nice visualisations
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Mapping, Tuple, Sequence, List
import scipy.linalg
from scipy.linalg import block_diag
from torch_geometric.utils import to_dense_adj

dataset = Planetoid(root='data/', name='cora', split='public')

def sym_norm_adj(A):
    #### Create the symmetric normalised adjacency from the dense adj matrix A
    A_tilde = A + torch.eye(A.shape[0])
    D_tilde = torch.diag(torch.sum(A_tilde, axis=1))
    D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
    A_tilde = A_tilde.to_sparse()
    D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
    adj_norm = torch.sparse.mm(torch.sparse.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
    return adj_norm

class SheafConvLayer(nn.Module):
    """A Sheaf Convolutional Network Layer with a learned sheaf.
        Args:
            num_nodes (int): Number of nodes in the graph
            input_dim (int): Dimensionality of the input feature vectors
            output_dim (int): Dimensionality of the output softmax distribution
            edge_index (torch.Tensor): Tensor of shape (2, num_edges)
    """
    def __init__(self, num_nodes, input_dim, output_dim, edge_index, step_size):
        super(SheafConvLayer, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_index = edge_index
        self.step_size = step_size
        self.linear = nn.Linear(input_dim, output_dim)
        self.sheaf_learner = nn.Linear(2*input_dim, 1, bias=False)
        self.left_idx, self.right_idx = self.compute_left_right_map_index()
        # This is only needed by our functions to compute Dirichlet energy
        # It should not be used below
        self.adj_norm = sym_norm_adj(to_dense_adj(edge_index)[0])

    def compute_left_right_map_index(self):
        """Computes indices for the full Laplacian matrix"""
        edge_to_idx = dict()
        for e in range(self.edge_index.size(1)):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            edge_to_idx[(source, target)] = e

        left_index, right_index = [], []
        row, col = [], []
        for e in range(self.edge_index.size(1)):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])
            row.append(source)
            col.append(target)

        left_index = torch.tensor(left_index, dtype=torch.long, device=self.edge_index.device)
        right_index = torch.tensor(right_index, dtype=torch.long, device=self.edge_index.device)
        left_right_index = torch.vstack([left_index, right_index])

        assert len(left_index) == edge_index.size(1)
        return left_right_index

    def build_laplacian(self, maps):
        """Builds the normalised Laplacian from the restriction maps.
        Args:
            maps: A tensor of shape (num_edges, 1) containing the scalar restriction map
                  for the source node of the respective edges in edge_index
            Returns Laplacian as a sparse COO tensor. 
        """
        # ================= Your code here ======================
        row, col = self.edge_index

        left_maps = torch.index_select(maps, index=self.left_idx, dim=0)
        right_maps = torch.index_select(maps, index=self.right_idx, dim=0)
        non_diag_maps = -left_maps * right_maps
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.num_nodes)

        d_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
        norm_maps = left_norm * non_diag_maps * right_norm
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv

        diag_indices = torch.arange(0, self.num_nodes, device=maps.device).view(1, -1).tile(2, 1)
        all_indices = torch.cat([diag_indices, self.edge_index], dim=-1)
        all_values = torch.cat([diag.view(-1), norm_maps.view(-1)])
        return torch.sparse_coo_tensor(all_indices, all_values, size=(self.num_nodes, self.num_nodes))

    def predict_restriction_maps(self, x):
        """Builds the normalised Laplacian from the restriction maps.
        Args:
            maps: A tensor of shape (num_edges, 1) containing the scalar restriction map
                  for the source node of the respective edges in edge_index
            Returns Laplacian as a sparse COO tensor. 
        """
        # ================= Your code here ======================
        row, col = self.edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.sheaf_learner(torch.cat([x_row, x_col], dim=1))
        maps = torch.tanh(maps)  
        return maps

    def forward(self, x):
        maps = self.predict_restriction_maps(x)
        laplacian = self.build_laplacian(maps)
        self.sheaf_laplacian = laplacian
        y = self.linear(x)
        #print(x.shape, laplacian.shape, y.shape)
        x = x - self.step_size * torch.sparse.mm(laplacian, y)
        return x

class SheafNN(nn.Module):
    """Simple encoder decoder GNN model using the various conv layers implemented by students

    Args:
        num_nodes (int): The number of nodes in the graph
        input_dim (int): Dimensionality of the input feature vectors
        hidden_dim (int): Dimensionality of the hidden feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        time (int):
        step_size (int):
        edge_index (torch.Tensor)
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, T, step_size, edge_index):
        super(SheafNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = int(T // step_size)

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.conv_layer = SheafConvLayer(num_nodes, hidden_dim, hidden_dim, edge_index, step_size)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.evolution = []

    def forward(self, x):
        self.evolution = []
        x = self.encoder(x)
        for _ in range(self.layers):
            self.evolution.append(x)
            x = self.conv_layer(x) #note implicitly we are sharing weights by using 1 conv block repeated
        self.evolution.append(x)
        #self.sheaf_laplacian = self.conv_layer.sheaf_laplacian
        x = self.decoder(x)
        y_hat = F.log_softmax(x, dim=1)
        return y_hat

data = dataset.data
#Cora  = CoraDataset()
A = torch.zeros((data.num_nodes, data.num_nodes))
A[data.edge_index[0], data.edge_index[1]] = 1
edge_index = data.edge_index
edge_index = remove_self_loops(edge_index)[0]
X = data.x

model = SheafNN(X.size(0), input_dim=X.shape[-1], hidden_dim=32, 
                output_dim=7, T=3, step_size=1., edge_index=edge_index)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

def train_sheaf(data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.  
    pred = out.argmax(dim=1)
    # training acc
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = train_correct.sum().item() / data.train_mask.sum().item()
    # validation acc
    valid_correct = pred[data.val_mask] == data.y[data.val_mask] 
    valid_acc = valid_correct.sum().item() / data.val_mask.sum().item()
    # test acc
    test_correct = pred[data.test_mask] == data.y[data.test_mask] 
    test_acc = valid_correct.sum().item() / data.val_mask.sum().item()

    print(f'train acc = {train_acc}, val acc = {valid_acc}, test acc = {test_acc}')
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    #return loss, train_acc, h

if __name__ == '__main__':
    print('SDN')
    for epoch in range(200):
        train_sheaf(dataset.data)
    