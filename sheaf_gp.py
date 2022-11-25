import numpy as np
import math
import itertools
import scipy as sp
import random
import tensorflow as tf
from torch_geometric import datasets
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import positive, print_summary

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", default='Cora', type=str, help="Cora, Citeseer, Texas, Wisconsin, Cornell, Chameleon, Squirrel")
parser.add_argument("--base_kernel", default='Polynomial', type=str, help="Polynomial, Matern52, Matern32, Matern12, SquaredPolynomial")
parser = parser.parse_args()

dataset_name = parser.data
dataset_path = f'data/'
if dataset_name in ["Cora", "Citeseer", "PubMed"]:
    dataset = datasets.Planetoid(dataset_path, dataset_name)
elif dataset_name in ["Computers", "Photo"]:
    dataset = datasets.Amazon(dataset_path, dataset_name)
elif dataset_name in ["Physics", "CS"]:
    dataset = datasets.Coauthor(dataset_path, dataset_name)
elif dataset_name in ["Texas", "Cornell", "Wisconsin"]:
    dataset = datasets.WebKB(dataset_path, dataset_name)
elif dataset_name in ["Chameleon", "Squirrel"]:
    dataset = datasets.WikipediaNetwork(dataset_path, dataset_name)
elif dataset_name in ["Actor"]:
    dataset = datasets.Actor(dataset_path, dataset_name)
data = dataset.data

# use first mask if there are multiple
try:
    data.train_mask.size(1)
    data.train_mask, data.val_mask, data.test_mask = data.train_mask[:,0], data.val_mask[:,0], data.test_mask[:,0]
except:
    pass

from torch_geometric.utils import remove_self_loops
data.edge_index = remove_self_loops(data.edge_index)[0]

if parser.base_kernel == 'Polynomial':
    base_kernel = gpflow.kernels.Polynomial()
elif parser.base_kernel == 'Matern12':
    base_kernel = gpflow.kernels.Matern12()
elif parser.base_kernel == 'Matern32':
    base_kernel = gpflow.kernels.Matern32()
elif parser.base_kernel == 'Matern52':
    base_kernel = gpflow.kernels.Matern52()
elif parser.base_kernel == 'SquaredExponential':
    base_kernel = gpflow.kernels.SquaredExponential()
'''
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='data/', name='cora', split='public')
data = dataset.data
'''
from kernels import Sheaf, SheafGGP

def step_callback(step, variables=None, values=None):
    pred = tf.math.argmax(m.predict_f(tf.cast(np.where(data.test_mask)[0].reshape(-1,1), dtype = tf.float64))[0], axis = 1)
    correct = np.sum(pred == data.y[data.test_mask])
    test_acc = 100.*correct/np.sum(data.test_mask.numpy())
    pred = tf.math.argmax(m.predict_f(tf.cast(np.where(data.val_mask)[0].reshape(-1,1), dtype = tf.float64))[0], axis = 1)
    correct = np.sum(pred == data.y[data.val_mask])
    val_acc = 100.*correct/np.sum(data.val_mask.numpy())
    print('Epoch = {}, val acc = {:.2f}, acc = {:.2f}'.format(step, val_acc, test_acc))
    #print_summary(m)

def optimize_tf(model, step_callback, lr=0.01):
    opt = tf.optimizers.Adam(lr=lr)
    for epoch_idx in range(200):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            loss = model.training_loss()
            gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        step_callback(epoch_idx)
        
if __name__ == '__main__':
    kernel = SheafGGP(data, base_kernel=base_kernel)
    n_class = data.y.numpy().max()+1
    invlink = gpflow.likelihoods.RobustMax(n_class)  # Robustmax inverse link function
    likelihood = gpflow.likelihoods.MultiClass(n_class, invlink=invlink)  # Multiclass likelihood
    m = gpflow.models.VGP(
        (tf.cast(np.where(data.train_mask)[0].reshape(-1,1), dtype = tf.float64), tf.reshape(data.y[data.train_mask], (-1,1))),
        likelihood=likelihood, 
        kernel=kernel, 
        num_latent_gps=n_class
    )
    #print_summary(m)

    optimize_tf(m, step_callback, lr = 0.1)