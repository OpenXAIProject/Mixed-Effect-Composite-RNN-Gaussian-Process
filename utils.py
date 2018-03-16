#Copyright 2018 KAIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License"); 
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import numpy as np
from layers import *


"""
These functions are for the GP class and data generation.
Only functions related to data generation is in use and others are not needed anymore,
since they are replaced by GPflow.
"""


# Define Kernel (covariance matrix)
def Kernel(X1, X2, t0, t1):
    sqdist = square_dist(X1, X2, t1)
    return t0**2 * np.exp(-.5 * sqdist)


# Description: Data generation from GP specified by W, theta's
# 'N' is a number of data points in one dimension line.
def data_from_GP(params, N, low=-5., high=5., dim=1):
    # Parameters defining generation model
    W, B, t0, t1 = params['W'], params['B'], params['t0'], params['t1']
    
    # Dimension check
    assert (dim == len(W)-1) and (dim == B.shape[0]-1 == B.shape[1]-1), 'Dimension match error'

    # Define input data points
    X = [np.linspace(low, high, N) for _ in range(dim)]
    X = np.meshgrid(*X)
    D_X = np.empty((N**dim, 0))
    for i in range(dim):
        D_X = np.concatenate((D_X, X[i].reshape(-1, 1)), axis=1)

    # Generate y values
    D_X_aug = np.concatenate((np.ones((N**dim, 1)), D_X), axis=1)
    K = Kernel(D_X, D_X, t0, t1)
    L = np.linalg.cholesky(K + 1e-12*np.eye(N**dim))
    U = np.random.normal(size=(N**dim, 1))
    y = np.matmul(D_X_aug, W) + np.matmul(L, U)

    K_noisy = Kernel(D_X, D_X, t0, t1) + np.dot(np.dot(D_X_aug, B), D_X_aug.T)
    L_noisy = np.linalg.cholesky(K_noisy + 1e-12*np.eye(N))
    y_noisy = np.matmul(D_X_aug, W) + np.matmul(L_noisy, U)

    if dim == 1:
        X = np.array(X).reshape(-1, 1)
        y = y.reshape(-1, 1)
    else:
        y = y.reshape([N for _ in range(dim)])
    return X, y, y_noisy, None


def simulation_data(num_data_each=100, num_patients=10, low=-5., high=5., \
                    input_dim=1, hidden_dim=5, output_dim=1, ratio=0.5, classification=False):
    """
    The data generation for simulating our model.
    The data generation model follows the model where all GPs(patients) share MLP mean function, 
    and each has its own covariance function.
    Input points for each GP might be different.
    MLP is one hideen layer model.

    Args:
        num_data_each: Number of data for each patient, N.
        num_patients: Number of paitents, P.
        low: Low limit value of range of input, x.
        high: High limit value of range of input, x.
        input_dim: Dimension of inputs.
        hidden_dim: Dimension of hiddens.
        output_dim: Dimension of outputs.
        ratio: Ratio between training data and testing data.
    """
    mu_W1 = 2*np.random.rand(input_dim*hidden_dim)-1
    mu_W2 = 2*np.random.rand(hidden_dim*output_dim)-1
    mu_b1 = 2*np.random.rand(hidden_dim)-1
    mu_b2 = 2*np.random.rand(output_dim)-1
    cov_W1 = 0.01*np.eye(input_dim*hidden_dim)
    cov_W2 = 0.01*np.eye(hidden_dim*output_dim)
    cov_b1 = 0.01*np.eye(hidden_dim)
    cov_b2 = 0.01*np.eye(output_dim)

    N = num_data_each*num_patients
    X = np.random.uniform(low=low, high=high, size=N).reshape(-1, 1)

    sample_W1 = np.random.multivariate_normal(mu_W1, cov_W1, N).reshape(N, input_dim, hidden_dim)
    sample_W2 = np.random.multivariate_normal(mu_W2, cov_W2, N).reshape(N, hidden_dim, output_dim)
    sample_b1 = np.random.multivariate_normal(mu_b1, cov_b1, N).reshape(N, hidden_dim)
    sample_b2 = np.random.multivariate_normal(mu_b2, cov_b2, N).reshape(N, output_dim)
    
    M = np.zeros((N, 1))
    for i in range(N):
        M[i] = np.matmul(np.maximum(np.matmul(X[i].reshape(1, -1), sample_W1[i]) + sample_b1[i], 0), \
               sample_W2[i]) + sample_b2[i]

    data = {'X_train': np.empty((0, input_dim)), 'y_train': np.empty((0, output_dim)), \
            'X_test': np.empty((0, input_dim)), 'y_test': np.empty((0, output_dim))}
    for i in range(num_patients):
        N_each = num_data_each
        N_each_train = int(N_each*ratio)
        N_each_test = N_each - N_each_train
        t0, t1 = np.random.uniform(low=0.7, high=1.414, size=2)
        _X = X[i*N_each:(i+1)*N_each, :]

        K = Kernel(_X, _X, t0, t1)
        L = np.linalg.cholesky(K + 1e-12*np.eye(N_each))
        U = np.random.normal(size=(N_each, 1))
        y = M[i*N_each:(i+1)*N_each, :] + np.matmul(L, U)

        if classification:
            y_mean = np.mean(y)
            for j in range(N_each):
                if y[j] <= y_mean:
                    y[j] = 0
                else:
                    y[j] = 1
            #y_mean = np.mean(y)
            #y[y<=y_mean] = 0
            #y[y>y_mean] = 1

        idx_shuffled = np.arange(N_each)
        np.random.shuffle(idx_shuffled)

        data['X_'+str(i+1)+'_train'] = _X[idx_shuffled[0:N_each_train]]
        data['y_'+str(i+1)+'_train'] = y[idx_shuffled[0:N_each_train]]
        data['X_'+str(i+1)+'_test'] = _X[idx_shuffled[N_each_train:N_each]]
        data['y_'+str(i+1)+'_test'] = y[idx_shuffled[N_each_train:N_each]]

        data['X_train'] = np.append(data['X_train'], data['X_'+str(i+1)+'_train'], axis=0)
        data['y_train'] = np.append(data['y_train'], data['y_'+str(i+1)+'_train'], axis=0)
        data['X_test'] = np.append(data['X_test'], data['X_'+str(i+1)+'_test'], axis=0)
        data['y_test'] = np.append(data['y_test'], data['y_'+str(i+1)+'_test'], axis=0)

    data['X'] = X
    data['M'] = M

    return data


def simulation_data_for_mtgp(nit, num_data_each=100, num_patients=10, low=-5., high=5., \
                             input_dim=1, hidden_dim=5, output_dim=1, ratio=0.5, classification=False):
    """
    The data generation for simulating our model.
    The data generation model follows the model where all GPs(patients) share MLP mean function, 
    and each has its own covariance function.
    Input points for each GP might be different.
    MLP is one hideen layer model.

    Args:
        num_data_each: Number of data for each patient, N.
        num_patients: Number of paitents, P.
        low: Low limit value of range of input, x.
        high: High limit value of range of input, x.
        input_dim: Dimension of inputs.
        hidden_dim: Dimension of hiddens.
        output_dim: Dimension of outputs.
        ratio: Ratio between training data and testing data.
    """
    mu_W1 = 2*np.random.rand(input_dim*hidden_dim)-1
    mu_W2 = 2*np.random.rand(hidden_dim*output_dim)-1
    mu_b1 = 2*np.random.rand(hidden_dim)-1
    mu_b2 = 2*np.random.rand(output_dim)-1
    cov_W1 = 0.01*np.eye(input_dim*hidden_dim)
    cov_W2 = 0.01*np.eye(hidden_dim*output_dim)
    cov_b1 = 0.01*np.eye(hidden_dim)
    cov_b2 = 0.01*np.eye(output_dim)

    N = num_data_each*num_patients
    _X = np.random.uniform(low=low, high=high, size=num_data_each).reshape(-1, 1)
    _X = np.sort(_X, axis=0)
    X = np.copy(_X)
    for i in range(num_patients-1):
        X = np.append(X, _X, axis=0)

    sample_W1 = np.random.multivariate_normal(mu_W1, cov_W1, N).reshape(N, input_dim, hidden_dim)
    sample_W2 = np.random.multivariate_normal(mu_W2, cov_W2, N).reshape(N, hidden_dim, output_dim)
    sample_b1 = np.random.multivariate_normal(mu_b1, cov_b1, N).reshape(N, hidden_dim)
    sample_b2 = np.random.multivariate_normal(mu_b2, cov_b2, N).reshape(N, output_dim)
    
    M = np.zeros((N, 1))
    for i in range(N):
        M[i] = np.matmul(np.maximum(np.matmul(X[i].reshape(1, -1), sample_W1[i]) + sample_b1[i], 0), \
               sample_W2[i]) + sample_b2[i]

    data = {'X_train': np.empty((0, input_dim)), 'y_train': np.empty((0, output_dim)), \
            'X_test': np.empty((0, input_dim)), 'y_test': np.empty((0, output_dim))}
    for i in range(num_patients):
        N_each = num_data_each
        N_each_train = int(N_each*ratio)
        N_each_test = N_each - N_each_train
        t0, t1 = np.random.uniform(low=0.7, high=1.414, size=2)
        _X = X[i*N_each:(i+1)*N_each, :]

        K = Kernel(_X, _X, t0, t1)
        L = np.linalg.cholesky(K + 1e-12*np.eye(N_each))
        U = np.random.normal(size=(N_each, 1))
        y = M[i*N_each:(i+1)*N_each, :] + np.matmul(L, U)

        if classification:
            y_mean = np.mean(y)
            for j in range(N_each):
                if y[j] <= y_mean:
                    y[j] = 0
                else:
                    y[j] = 1
            #y_mean = np.mean(y)
            #y[y<=y_mean] = 0
            #y[y>y_mean] = 1

        idx_shuffled = np.arange(N_each)
        np.random.shuffle(idx_shuffled)

        data['X_'+str(i+1)+'_train'] = _X[idx_shuffled[0:N_each_train]]
        data['y_'+str(i+1)+'_train'] = y[idx_shuffled[0:N_each_train]]
        data['X_'+str(i+1)+'_test'] = _X[idx_shuffled[N_each_train:N_each]]
        data['y_'+str(i+1)+'_test'] = y[idx_shuffled[N_each_train:N_each]]

        data['X_train'] = np.append(data['X_train'], data['X_'+str(i+1)+'_train'], axis=0)
        data['y_train'] = np.append(data['y_train'], data['y_'+str(i+1)+'_train'], axis=0)
        data['X_test'] = np.append(data['X_test'], data['X_'+str(i+1)+'_test'], axis=0)
        data['y_test'] = np.append(data['y_test'], data['y_'+str(i+1)+'_test'], axis=0)

        data['y_'+str(i+1)] = y
        data['X_idx_'+str(i+1)] = np.array(idx_shuffled[0:N_each_train]).reshape(-1, 1)

    data['X'] = X
    data['M'] = M

    N_train = int(100*ratio)
    X = _X
    xtrain = _X
    Y = np.empty((100, 0))
    ytrain = np.empty((0, 1))
    ind_kf_train = np.empty((0, 1))
    ind_kx_train = np.empty((0, 1))
    for i in range(1, num_patients+1):
        Y = np.append(Y, data['y_'+str(i)], axis=1)
        ytrain = np.append(ytrain, data['y_'+str(i)+'_train'], axis=0)
        ind_kf_train = np.append(ind_kf_train, np.zeros((N_train, 1))+i, axis=0)
        ind_kx_train = np.append(ind_kx_train, data['X_idx_'+str(i)]+1, axis=0)
    nx = np.ones((N_train*num_patients, 1))

    assert X.shape == (100, 1)
    assert Y.shape == (100, num_patients)
    assert xtrain.shape == (100, 1)
    assert ytrain.shape == (N_train*num_patients, 1)
    assert ind_kf_train.shape == (N_train*num_patients, 1)
    assert ind_kx_train.shape == (N_train*num_patients, 1)
    assert nx.shape == (N_train*num_patients, 1)

    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/X_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', X, delimiter=',')
    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/Y_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', Y, delimiter=',')
    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/xtrain_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', xtrain, delimiter=',')
    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/ytrain_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', ytrain, delimiter=',')
    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/ind_kf_train_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', ind_kf_train, delimiter=',')
    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/ind_kx_train_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', ind_kx_train, delimiter=',')
    np.savetxt('/home/user/Dropbox/research/mtgp-master/data/nx_'+str(num_patients)+'_'+str(ratio)+'_'+str(nit)+'.csv', nx, delimiter=',')

    return data



# Description: zero mean standard GP prediction
def GP_zero_mean(X_train, y_train, X_test, N_test, params):
    t0 = params['t0']
    t1 = params['t1']

    # Kernel matrices
    K = Kernel(X_train, X_train, t0, t1)
    K_inv = np.linalg.inv(K)
    K_s = Kernel(X_train, X_test, t0, t1)
    K_ss = Kernel(X_test, X_test, t0, t1)

    # Std for plotting
    L = np.linalg.cholesky(K + 1e-12*np.eye(len(X_train)))
    Lk = np.linalg.solve(L, K_s)
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)

    mu = np.dot(Lk.T, np.linalg.solve(L, y_train)).reshape(N_test, -1)
    KK = K_ss - np.dot(Lk.T, Lk)
    L = np.linalg.cholesky(KK + 1e-12*np.eye(N_test))

    # Sampling from GP
    U = np.random.normal(size=(N_test, 1))
    y_test = mu + np.matmul(L, U)

    return y_test, mu, stdv.reshape(N_test, -1)


# Description: Linear mean GP prediction
def GP_linear_mean(X_train, y_train, X_test, N_test, params):
    t0 = params['t0']
    t1 = params['t1']
    W = params['W']

    # Define mean function
    def mean(W, X):
        N, D = X.shape
        return np.matmul(np.concatenate((np.ones((N, 1)), X), axis=1), W)

    # Kernel matrices
    K = Kernel(X_train, X_train, t0, t1)
    K_inv = np.linalg.inv(K)
    K_s = Kernel(X_train, X_test, t0, t1)
    K_ss = Kernel(X_test, X_test, t0, t1)

    # Std for plotting
    L = np.linalg.cholesky(K + 1e-12*np.eye(len(X_train)))
    Lk = np.linalg.solve(L, K_s)
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)

    m_X_test = mean(W, X_test)
    m_X_train = mean(W, X_train)
    mu = m_X_test + np.dot(Lk.T, np.linalg.solve(L, y_train-m_X_train)).reshape(N_test, -1)
    KK = K_ss - np.dot(Lk.T, Lk)
    L = np.linalg.cholesky(KK + 1e-12*np.eye(N_test))

    # Sampling from GP
    U = np.random.normal(size=(N_test, 1))
    y_test = mu + np.matmul(L, U)

    return y_test, mu, stdv.reshape(N_test, -1)


# Description: MLP mean GP prediction
def GP_MLP_mean(X_train, y_train, X_test, N_test, params):
    t0 = params['t0']
    t1 = params['t1']
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']

    # Define mean function
    def mean(W1, W2, b1, b2, X):
        h, cache = affine_relu_forward(X, W1, b1)
        o, cache2 = affine_forward(h, W2, b2)
        return o

    # Kernel matrices
    K = Kernel(X_train, X_train, t0, t1)
    K_inv = np.linalg.inv(K)
    K_s = Kernel(X_train, X_test, t0, t1)
    K_ss = Kernel(X_test, X_test, t0, t1)

    # Std for plotting
    L = np.linalg.cholesky(K + 1e-12*np.eye(len(X_train)))
    Lk = np.linalg.solve(L, K_s)
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)

    m_X_test = mean(W1, W2, b1, b2, X_test)
    m_X_train = mean(W1, W2, b1, b2, X_train)
    mu = m_X_test + np.dot(Lk.T, np.linalg.solve(L, y_train-m_X_train)).reshape(N_test, -1)
    KK = K_ss - np.dot(Lk.T, Lk)
    L = np.linalg.cholesky(KK + 1e-12*np.eye(N_test))

    # Sampling from GP
    U = np.random.normal(size=(N_test, 1))
    y_test = mu + np.matmul(L, U)

    return y_test, mu, stdv.reshape(N_test, -1)


# Compute 2-norm (sqdist)
def square_dist(X1, X2, t1):
    _X1 = X1*t1
    _X2 = X2*t1
    return np.sum(_X1**2,1).reshape(-1,1) + np.sum(_X2**2,1) - 2*np.dot(_X1, _X2.T)


# Compute K and its inverse
def K_inverse(X, sqdist, params):
    K = params['t0']**2 * np.exp(-.5 * sqdist)
    return K, np.linalg.inv(K)


# Compute B0: derivative by t0
def dK_dt0(sqdist, params):
    return 2*params['t0']*np.exp(-.5 * sqdist)


# Compute B1: derivative by t1
def dK_dt1(sqdist, params):
    return params['t0']**2 * np.exp(-.5 * sqdist) * (-params['t1'])**3 * sqdist


# Compute B2: derivative by t2
def dK_dt2(N_train):
    return np.ones((N_train, N_train))


# Compute B3: derivative by t3
def dK_dt3(X):
    return np.matmul(X, X.T)


# Compute B4: derivative by W
def dL_dW(X, W, K_inv, y):
    return (np.matmul(np.matmul(y.T, K_inv), X) - np.matmul(np.matmul(np.matmul(W.T, X.T), K_inv), X)).T


# Derivative of log likelihood
def dL_dt(A, B):
    N, M = A.shape
    toReturn = 0.
    for i in range(N):
        for j in range(N):
            toReturn += A[i, j]*B[j, i]
    return .5 * toReturn


# Log likelihood function
def log_likelihood(K, K_inv, y_train):
    return (-.5*np.linalg.det(K)) - (.5*np.matmul(y_train.T, np.matmul(K_inv, y_train)))


