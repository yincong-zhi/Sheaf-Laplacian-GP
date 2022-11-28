import numpy as np
import math
import itertools
import scipy as sp
import random
import tensorflow as tf
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import positive, print_summary

class SheafGGP(gpflow.kernels.Kernel):
    def __init__(self, data, alpha = 1., base_kernel = gpflow.kernels.Polynomial()):
        super().__init__()
        self.alpha = gpflow.Parameter(alpha, transform = positive())
        #self.variance = gpflow.Parameter(variance, transform = positive())
        self.base_kernel = base_kernel
        self.node_feats = data.x
        self.edge_index = data.edge_index
        self.num_nodes = data.num_nodes

        input_dim = data.x.shape[-1]
        tf.keras.backend.set_floatx('float64')
        initializer = None#tf.keras.initializers.Ones()
        sheaf_learner = tf.keras.models.Sequential()
        sheaf_learner.add(tf.keras.Input(shape=(2*input_dim,)))
        sheaf_learner.add(tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=initializer))
        self.sheaf_learner = sheaf_learner
    
    def sheaf_construct(self, sheaf_learner = None):
        sheaf_learner = self.sheaf_learner if sheaf_learner is None else sheaf_learner
        # index each edge
        edge_to_idx = dict()
        for e in range(self.edge_index.shape[1]):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            edge_to_idx[(source, target)] = e

        # index of each edge
        left_index, right_index = [], []
        for e in range(self.edge_index.shape[1]):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            left_index.append(e)
            right_index.append(edge_to_idx[(source, target)])

        #left_index = torch.tensor(left_index, dtype=torch.long, device=edge_index.device)
        #right_index = torch.tensor(right_index, dtype=torch.long, device=edge_index.device)
        left_right_index = tf.stack([left_index, right_index])

        x = self.node_feats
        # define perceptron on 
        row, col = self.edge_index
        x_row = tf.gather(x, row)
        x_col = tf.gather(x, col)
        maps = sheaf_learner(tf.concat([x_row, x_col], axis=1))
        #maps = tf.tanh(maps)
        #maps = tf.keras.activations.tanh(maps)

        #left_maps = torch.index_select(maps, index=left_index, dim=0)
        left_maps = tf.gather(maps, left_index)
        #right_maps = torch.index_select(maps, index=right_index, dim=0)
        right_maps = tf.gather(maps, right_index)
        non_diag_maps = -left_maps * left_maps
        # reduce to size N (size of the graph)
        ref = tf.Variable(tf.zeros(self.num_nodes, dtype = tf.float64))
        indices = tf.constant(row.reshape(-1,1))
        updates = maps**2
        diag_maps = tf.compat.v1.scatter_add(ref, indices, updates)

        d_sqrt_inv = tf.math.pow(diag_maps + 1, -0.5)
        left_norm, right_norm = tf.gather(d_sqrt_inv,row.numpy()), tf.gather(d_sqrt_inv,col.numpy())
        norm_maps = tf.reshape(left_norm, (-1,1)) * non_diag_maps * tf.reshape(right_norm, (-1,1))
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv
        diag = tf.reshape(diag, (-1,1))

        diag_indices = tf.stack((tf.range(self.num_nodes), tf.range(self.num_nodes)))
        all_indices = tf.concat([diag_indices, self.edge_index], axis=1)
        all_values = tf.concat([tf.reshape(diag, -1), tf.reshape(norm_maps, -1)], axis=0)
        laplacian = tf.sparse.SparseTensor(tf.cast(tf.transpose(all_indices), dtype=tf.int64), all_values, dense_shape=(self.num_nodes, self.num_nodes))
        laplacian = tf.sparse.reorder(laplacian)
        #laplacian = tf.sparse.to_dense(laplacian)
        return laplacian

    def K(self, X, X2=None):
        X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32) if X2 is not None else X
        inner_cov = self.base_kernel.K(self.node_feats.numpy())
        S = tf.linalg.inv(tf.eye(self.num_nodes, dtype = tf.float64) + self.alpha * tf.sparse.to_dense(self.sheaf_construct(self.sheaf_learner)))
        #S = tf.linalg.expm(- self.alpha * self.sheaf_construct(self.sheaf_learner))
        #S = tf.eye(self.num_nodes, dtype = tf.float64) - self.alpha * self.sheaf_construct(self.sheaf_learner)
        total_cov = S @ inner_cov @ tf.transpose(S)
        cov = tf.gather(total_cov, X, axis=0)
        cov = tf.gather(cov, X2, axis=1)
        return cov

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))
        
class Sheaf(gpflow.kernels.Kernel):
    def __init__(self, data, alpha = 1., variance = 1.):
        super(Sheaf, self).__init__()
        self.alpha = gpflow.Parameter(alpha, transform = positive())
        self.variance = gpflow.Parameter(variance, transform = positive())
        self.x = data.x
        self.edge_index = data.edge_index
        self.num_nodes = data.num_nodes
        input_dim = data.x.shape[-1]
        tf.keras.backend.set_floatx('float64')
        initializer = None#tf.keras.initializers.Ones()
        sheaf_learner = tf.keras.models.Sequential()
        sheaf_learner.add(tf.keras.Input(shape=(2*input_dim,)))
        sheaf_learner.add(tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=initializer))
        self.sheaf_learner = sheaf_learner
    
    def sheaf_construct(self, sheaf_learner = None):
        sheaf_learner = self.sheaf_learner if sheaf_learner is None else sheaf_learner
        # index each edge
        edge_to_idx = dict()
        for e in range(self.edge_index.shape[1]):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            edge_to_idx[(source, target)] = e

        # index of each edge
        left_index, right_index = [], []
        for e in range(self.edge_index.shape[1]):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

        #left_index = torch.tensor(left_index, dtype=torch.long, device=edge_index.device)
        #right_index = torch.tensor(right_index, dtype=torch.long, device=edge_index.device)
        left_right_index = tf.stack([left_index, right_index])

        x = self.x
        # define perceptron on 
        row, col = self.edge_index
        x_row = tf.gather(x, row)
        x_col = tf.gather(x, col)
        maps = sheaf_learner(tf.concat([x_row, x_col], axis=1))
        #maps = tf.keras.activations.tanh(maps)

        #left_maps = torch.index_select(maps, index=left_index, dim=0)
        left_maps = tf.gather(maps, left_index)
        #right_maps = torch.index_select(maps, index=right_index, dim=0)
        right_maps = tf.gather(maps, right_index)
        non_diag_maps = -left_maps * left_maps
        # reduce to size N (size of the graph)
        ref = tf.Variable(tf.zeros(self.num_nodes, dtype = tf.float64))
        indices = tf.constant(row.reshape(-1,1))
        updates = maps**2
        diag_maps = tf.compat.v1.scatter_add(ref, indices, updates)

        d_sqrt_inv = tf.math.pow(diag_maps + 1, -0.5)
        left_norm, right_norm = tf.gather(d_sqrt_inv,row.numpy()), tf.gather(d_sqrt_inv,col.numpy())
        norm_maps = tf.reshape(left_norm, (-1,1)) * non_diag_maps * tf.reshape(right_norm, (-1,1))
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv
        diag = tf.reshape(diag, (-1,1))

        diag_indices = tf.stack((tf.range(self.num_nodes), tf.range(self.num_nodes)))
        all_indices = tf.concat([diag_indices, self.edge_index], axis=1)
        all_values = tf.concat([tf.reshape(diag, -1), tf.reshape(norm_maps, -1)], axis=0)
        laplacian = tf.sparse.SparseTensor(tf.cast(tf.transpose(all_indices), dtype=tf.int64), all_values, dense_shape=(self.num_nodes, self.num_nodes))
        laplacian = tf.sparse.reorder(laplacian)
        #laplacian = tf.sparse.to_dense(laplacian)
        return laplacian

    def K(self, X, X2=None):
        X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32) if X2 is not None else X
        total_cov = self.variance * tf.linalg.inv(tf.eye(self.num_nodes, dtype = tf.float64) + self.alpha * tf.sparse.to_dense(self.sheaf_construct(self.sheaf_learner)))
        #total_cov = self.variance * tf.linalg.expm(- self.alpha * self.sheaf_construct(self.sheaf_learner))
        cov = tf.gather(total_cov, X, axis=0)
        cov = tf.gather(cov, X2, axis=1)
        return cov

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))