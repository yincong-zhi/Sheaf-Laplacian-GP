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

dataset = Planetoid(root='data/', name='cora', split='public')
data = dataset.data
A = np.zeros((data.num_nodes, data.num_nodes))
A[data.edge_index[0], data.edge_index[1]] = 1
edge_index = data.edge_index
#edge_index = remove_self_loops(edge_index)[0]
num_nodes = data.num_nodes

class Sheaf(gpflow.kernels.Kernel):
    def __init__(self, alpha = 1., variance = 1.):
        super(Sheaf, self).__init__()
        self.alpha = gpflow.Parameter(alpha, transform = positive())
        self.variance = gpflow.Parameter(variance, transform = positive())
        input_dim = data.x.shape[-1]
        tf.keras.backend.set_floatx('float64')
        initializer = None#tf.keras.initializers.Ones()
        sheaf_learner = tf.keras.models.Sequential()
        sheaf_learner.add(tf.keras.Input(shape=(2*input_dim,)))
        sheaf_learner.add(tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer))
        self.sheaf_learner = sheaf_learner
    
    def sheaf_construct(self, sheaf_learner = None):
        sheaf_learner = self.sheaf_learner if sheaf_learner is None else sheaf_learner
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
        ref = tf.Variable(tf.zeros(num_nodes, dtype = tf.float64))
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
        return laplacian

    def K(self, X, X2=None):
        X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32) if X2 is not None else X
        total_cov = self.variance * tf.linalg.inv(tf.eye(num_nodes, dtype = tf.float64) + self.alpha * self.sheaf_construct(self.sheaf_learner))
        #total_cov = self.variance * tf.linalg.expm(- self.alpha * self.sheaf_construct(self.sheaf_learner))
        cov = tf.gather(total_cov, X, axis=0)
        cov = tf.gather(cov, X2, axis=1)
        return cov

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

if __name__ == '__main__':
    kernel = Sheaf(variance = 10.)
    n_class = data.y.numpy().max()+1
    invlink = gpflow.likelihoods.RobustMax(n_class)  # Robustmax inverse link function
    likelihood = gpflow.likelihoods.MultiClass(n_class, invlink=invlink)  # Multiclass likelihood
    m = gpflow.models.VGP(
        (tf.cast(np.where(data.train_mask)[0].reshape(-1,1), dtype = tf.float64), tf.reshape(data.y[data.train_mask], (-1,1))),
        likelihood=likelihood, 
        kernel=kernel, 
        num_latent_gps=n_class
    )
    print_summary(m)

    def step_callback(step, variables=None, values=None):
        if step % 1 == 0:
            pred = tf.math.argmax(m.predict_f(tf.cast(np.where(data.test_mask)[0].reshape(-1,1), dtype = tf.float64))[0], axis = 1)
            print('Epoch = {}, acc = {}'.format(step,np.sum(pred == data.y[data.test_mask])))
            #print_summary(m)

    def optimize_tf(model, step_callback, lr=0.01):
        opt = tf.optimizers.Adam(lr=lr)
        for epoch_idx in range(500):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                loss = model.training_loss()
                gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            step_callback(epoch_idx)

    optimize_tf(m, step_callback, lr = 0.1)