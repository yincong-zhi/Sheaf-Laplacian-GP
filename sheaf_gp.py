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

from kernels import Sheaf, SheafGGP

def step_callback(step, variables=None, values=None):
    if step % 1 == 0:
        pred = tf.math.argmax(m.predict_f(tf.cast(np.where(data.test_mask)[0].reshape(-1,1), dtype = tf.float64))[0], axis = 1)
        print('Epoch = {}, acc = {:.2f}'.format(step, 0.1*np.sum(pred == data.y[data.test_mask])))
        #print_summary(m)

def optimize_tf(model, step_callback, lr=0.01):
    opt = tf.optimizers.Adam(lr=lr)
    for epoch_idx in range(100):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            loss = model.training_loss()
            gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        step_callback(epoch_idx)
        
if __name__ == '__main__':
    kernel = SheafGGP(variance = 1.)
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

    optimize_tf(m, step_callback, lr = 0.1)