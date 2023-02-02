import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import positive

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
        sheaf_learner.add(tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer))
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
        maps = tf.keras.activations.relu(maps)

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

# A separate kernel is written for tanh activation as our local gpu 
# cannot evaluate tensorflow's tanh
class SheafGGP_t(gpflow.kernels.Kernel):
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
        sheaf_learner.add(tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer))
        self.sheaf_learner = sheaf_learner

    def tanh(self, x):
        return (tf.exp(2*x) - 1)/(tf.exp(2*x) + 1)
        
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
        maps = self.tanh(maps)

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

def chebyshev_polynomial(mat, coefficients):
    monomial_prev_exists = False        # Using these helper variables is required to make it work with the SciPy optimizer
    monomial_exists = False
    monomial_prev = tf.zeros_like(mat)
    monomial = tf.zeros_like(mat)
    polynomial = None
    for coeff in coefficients:
        if not monomial_exists:
            monomial = tf.eye(mat.shape[0], dtype=mat.dtype)
            monomial_exists = True
        elif not monomial_prev_exists:
            monomial_prev = monomial
            monomial = mat
            monomial_prev_exists = True
        else:
            temp = 2 * mat @ monomial - monomial_prev
            monomial_prev = monomial
            monomial = temp
        polynomial = coeff * monomial if polynomial is None else polynomial + coeff * monomial
    return polynomial

class SheafChebyshev(gpflow.kernels.base.Kernel):
    def __init__(self, poly_degree, node_feats, edge_index, base_kernel=None):
        super().__init__()
        self.num_nodes = node_feats.shape[0]
        #self.normalized_L = tf.cast(normalized_L, tf.float64) - tf.eye(self.num_nodes, dtype=tf.float64)
        self.coeffs = gpflow.Parameter(tf.ones(poly_degree+1))
        self.node_feats = tf.convert_to_tensor(node_feats, dtype = tf.float64)
        self.edge_index = edge_index
        self.base_kernel = base_kernel

        input_dim = node_feats.shape[-1]
        tf.keras.backend.set_floatx('float64')
        initializer = None#tf.keras.initializers.Ones()
        sheaf_learner = tf.keras.models.Sequential()
        sheaf_learner.add(tf.keras.Input(shape=(2*input_dim,)))
        sheaf_learner.add(tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer))
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

    def conv_mat(self):
        normalized_L = tf.sparse.to_dense(self.sheaf_construct(self.sheaf_learner)) - tf.eye(self.num_nodes, dtype=tf.float64)
        return chebyshev_polynomial(normalized_L, self.coeffs)

    def K(self, X, Y=None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        Y = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        if self.base_kernel is not None:
            cov = self.base_kernel.K(self.node_feats)
        else:
            cov = tf.eye(self.num_nodes, dtype=self.node_feats.dtype)
        conv_mat = self.conv_mat()
        cov = tf.matmul(conv_mat, tf.matmul(cov, tf.transpose(conv_mat)))
        return tf.gather(tf.gather(cov, X, axis=0), Y, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

