import numpy as np
import scipy as sp
import tensorflow as tf
from torch_geometric import datasets
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", default='Cora', type=str, help="Cora, Citeseer, Texas, Wisconsin, Cornell, Chameleon, Squirrel, Actor")
parser.add_argument("--base_kernel", default='Polynomial', type=str, help="Polynomial, Matern52, Matern32, Matern12, SquaredPolynomial")
parser.add_argument("--epoch", default=200, type=int, help="number of epochs")
parser.add_argument("--lr", default=0.1, type=float, help="adam learn rate")
parser.add_argument('--approx', type=bool, default=False, help='default is exact kernel, True for chebyshev approximation')
parser.add_argument('--approx_deg', type=int, default=7, help='degree of chebyshev approximation, only used when --approx=True')
parser.add_argument('--train_on_val', type=bool, default=False, help='if True, validation set is included in the training')
parser.add_argument('--split', type=int, default=0, help='data split if there are multiple')
parser.add_argument('--act', type=str, default='relu', help='relu, tanh')

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
    data.train_mask, data.val_mask, data.test_mask = data.train_mask[:,parser.split], data.val_mask[:,parser.split], data.test_mask[:,parser.split]
except:
    pass

if parser.train_on_val:
    data.train_mask += data.val_mask
    
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

from kernels import SheafGGP, SheafChebyshev, SheafGGP_t

def step_callback(step, variables=None, values=None):
    pred = tf.math.argmax(m.predict_f(tf.cast(np.where(data.test_mask)[0].reshape(-1,1), dtype = tf.float64))[0], axis = 1)
    correct = np.sum(pred == data.y[data.test_mask])
    test_acc = 100.*correct/np.sum(data.test_mask.numpy())
    pred = tf.math.argmax(m.predict_f(tf.cast(np.where(data.val_mask)[0].reshape(-1,1), dtype = tf.float64))[0], axis = 1)
    correct = np.sum(pred == data.y[data.val_mask])
    val_acc = 100.*correct/np.sum(data.val_mask.numpy())
    print('Epoch = {}, elbo = {:.2f}, val acc = {:.2f}, test acc = {:.2f}'.format(step, m.elbo().numpy(), val_acc, test_acc))
    #print_summary(m)

def optimize_tf(model, step_callback, lr=0.01):
    opt = tf.optimizers.Adam(lr=lr)
    elbos = []
    for epoch_idx in range(parser.epoch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            loss = model.training_loss()
            gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        step_callback(epoch_idx)
        elbos.append(model.elbo())
    #return elbos
        
if __name__ == '__main__':
    if parser.approx:
        kernel = SheafChebyshev(parser.approx_deg, data.x, data.edge_index, base_kernel)
    elif parser.act == 'relu':
        kernel = SheafGGP(data, base_kernel=base_kernel)
    elif parser.act == 'tanh':
        kernel = SheafGGP_t(data, base_kernel=base_kernel)
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

    optimize_tf(m, step_callback, lr = parser.lr)