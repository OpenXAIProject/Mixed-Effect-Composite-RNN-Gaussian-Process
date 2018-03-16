# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .param import Param, DataHolder
from .model import GPModel
from .mean_functions import Zero, TwoLayerSigmoidMLP, TwoLayerReLUMLP, MixtureExpertsMLP2, MixtureExpertsMLP4
from ._settings import settings
from . import transforms
from .conditionals import conditional
from .kullback_leiblers import gauss_kl
float_type = settings.dtypes.float_type


class VGP(GPModel):
    """
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    This implementation is equivalent to svgp with X=Z, but is more efficient.
    The whitened representation is used to aid optimization.

    The posterior approximation is

    .. math::

       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None, reg=0.0):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        """

        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.reg = reg

        self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
        q_sqrt = np.array([np.eye(self.num_data)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
        self.q_sqrt = Param(q_sqrt, transforms.LowerTriangular(self.num_data, self.num_latent))

    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_sqrt = Param(np.eye(self.num_data)[:, :, None] *
                                np.ones((1, 1, self.num_latent)))

        return super(VGP, self).compile(session=session,
                                        graph=graph,
                                        optimizer=optimizer)

    def build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

        """

        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        K = self.kern.K(self.X) + tf.eye(self.num_data, dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(K)

        fmean = tf.matmul(L, self.q_mu) + self.mean_function(self.X)  # NN,ND->ND

        q_sqrt_dnn = tf.matrix_band_part(tf.transpose(self.q_sqrt, [2, 0, 1]), -1, 0)  # D x N x N

        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent, 1, 1]))

        LTA = tf.matmul(L_tiled, q_sqrt_dnn)  # D x N x N
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        
        # Regularization if mean function is two layer MLP
        if isinstance(self.mean_function, TwoLayerSigmoidMLP) or isinstance(self.mean_function, TwoLayerReLUMLP):
            reg = tf.reduce_sum(tf.square(self.mean_function.W1)) + tf.reduce_sum(tf.square(self.mean_function.W2))
        elif isinstance(self.mean_function, MixtureExpertsMLP2):
            reg =tf.reduce_sum(tf.square(self.mean_function.W1_1))+tf.reduce_sum(tf.square(self.mean_function.W1_2))+\
                 tf.reduce_sum(tf.square(self.mean_function.W2_1))+tf.reduce_sum(tf.square(self.mean_function.W2_2))+\
                 tf.reduce_sum(tf.square(self.mean_function.W))
        elif isinstance(self.mean_function, MixtureExpertsMLP4):
            reg =tf.reduce_sum(tf.square(self.mean_function.W1_1))+tf.reduce_sum(tf.square(self.mean_function.W1_2))+\
                 tf.reduce_sum(tf.square(self.mean_function.W2_1))+tf.reduce_sum(tf.square(self.mean_function.W2_2))+\
                 tf.reduce_sum(tf.square(self.mean_function.W3_1))+tf.reduce_sum(tf.square(self.mean_function.W3_2))+\
                 tf.reduce_sum(tf.square(self.mean_function.W4_1))+tf.reduce_sum(tf.square(self.mean_function.W4_2))+\
                 tf.reduce_sum(tf.square(self.mean_function.W))
        else:
            reg = 0.0

        return tf.reduce_sum(var_exp) - KL - (.5 * self.reg * reg)

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew), var


class VGP_RNN(GPModel):
    """
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    This implementation is equivalent to svgp with X=Z, but is more efficient.
    The whitened representation is used to aid optimization.

    The posterior approximation is

    .. math::

       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None, reg=0.0): # modified
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        """

        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.reg = reg # modified

        self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
        q_sqrt = np.array([np.eye(self.num_data)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
        self.q_sqrt = Param(q_sqrt, transforms.LowerTriangular(self.num_data, self.num_latent))

    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_sqrt = Param(np.eye(self.num_data)[:, :, None] *
                                np.ones((1, 1, self.num_latent)))

        return super(VGP_RNN, self).compile(session=session,
                                        graph=graph,
                                        optimizer=optimizer)

    def build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

        """

        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        K = self.kern.K(self.X) + tf.eye(self.num_data, dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(K)

        fmean = tf.matmul(L, self.q_mu) + self.mean_function(self.X)  # NN,ND->ND

        q_sqrt_dnn = tf.matrix_band_part(tf.transpose(self.q_sqrt, [2, 0, 1]), -1, 0)  # D x N x N

        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent, 1, 1]))

        LTA = tf.matmul(L_tiled, q_sqrt_dnn)  # D x N x N
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        
        # modified
        return tf.reduce_sum(var_exp) - KL - (.5 * self.reg * (tf.reduce_sum(tf.square(self.mean_function.Wemb)) + \
                                                               tf.reduce_sum(tf.square(self.mean_function.W)) + \
                                                               tf.reduce_sum(tf.square(self.mean_function.W2)) + \
                                                               tf.reduce_sum(tf.square(self.mean_function.Wout))))

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew, is_predict=True, X_train=self.X), var


class VGP_RNN_E2E(GPModel):
    """
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    This implementation is equivalent to svgp with X=Z, but is more efficient.
    The whitened representation is used to aid optimization.

    The posterior approximation is

    .. math::

       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None, reg=0.0): # modified
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        """

        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.reg = reg # regularization to weights parameters in mean function

        self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
        q_sqrt = np.array([np.eye(self.num_data)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
        self.q_sqrt = Param(q_sqrt, transforms.LowerTriangular(self.num_data, self.num_latent))

    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_sqrt = Param(np.eye(self.num_data)[:, :, None] *
                                np.ones((1, 1, self.num_latent)))

        return super(VGP_RNN_E2E, self).compile(session=session,
                                        graph=graph,
                                        optimizer=optimizer)

    def build_hidden_states(self, X, is_predict=False, X_train=None):
        """
        This method computes hidden states of RNN to feed them into GP
        """

        m = self.mean_function
        if is_predict:
            length = 1
            X_emb = tf.matmul(tf.concat([X_train, X], axis=0), m.Wemb)
        else:
            length = 0
            X_emb = tf.matmul(X, m.Wemb)

        # Feature representation of hidden states of RNN inspired by DKL
        H_emb = tf.matmul(X_emb, m.W[0:m.hidden_dim, :])
        H = []
        for i in range(m.length + length):
            if i == 0:
                #H.append(tf.tanh(H_emb[i, :]))
                #H.append(tf.maximum(H_emb[i, :], 0.0))
                H.append(tf.sigmoid(H_emb[i, :]))
            else:
                toAdd = tf.matmul(tf.reshape(H[i-1], [1, -1]), m.W[m.hidden_dim:2*m.hidden_dim, :]) + m.b
                #H.append(tf.tanh(H_emb[i, :] + toAdd[0, :]))
                #H.append(tf.maximum(H_emb[i, :] + toAdd[0, :], 0.0))
                H.append(tf.sigmoid(H_emb[i, :] + toAdd[0, :]))
        H = tf.stack(H, axis=0)
        
        if is_predict:
            return tf.reshape(H[-length:, :], [length, -1])
        else:
            return H

    def build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

        """

        # Feature representation of hidden states of RNN inspired by DKL
        H = self.build_hidden_states(self.X)

        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        #settings.numerics.jitter_level = 1e-2
        #print (settings.numerics.jitter_level)
        K = self.kern.K(H) + tf.eye(self.num_data, dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(K)

        fmean = tf.matmul(L, self.q_mu) + self.mean_function(H)  # NN,ND->ND

        q_sqrt_dnn = tf.matrix_band_part(tf.transpose(self.q_sqrt, [2, 0, 1]), -1, 0)  # D x N x N

        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent, 1, 1]))

        LTA = tf.matmul(L_tiled, q_sqrt_dnn)  # D x N x N
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        
        # modified
        return tf.reduce_sum(var_exp) - KL - (.5 * self.reg * (tf.reduce_sum(tf.square(self.mean_function.Wemb)) + \
                                                               tf.reduce_sum(tf.square(self.mean_function.W)) + \
                                                               tf.reduce_sum(tf.square(self.mean_function.W2)) + \
                                                               tf.reduce_sum(tf.square(self.mean_function.Wout))))

    def build_predict(self, Xnew, full_cov=False):
        Hnew = tf.cond(tf.reduce_all(tf.equal(self.X, Xnew)), lambda: self.build_hidden_states(Xnew), \
                       lambda: self.build_hidden_states(Xnew, is_predict=True, X_train=self.X))
        H = self.build_hidden_states(self.X)
        mu, var = conditional(Hnew, H, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        mean_val = tf.cond(tf.reduce_all(tf.equal(self.X, Xnew)), lambda: self.mean_function(Hnew), \
                           lambda: self.mean_function(Hnew, is_predict=True, X_train=H))
        return mu + mean_val, var


class VGP_opper_archambeau(GPModel):
    """
    This method approximates the Gaussian process posterior using a multivariate Gaussian.
    The key reference is:
    ::
      @article{Opper:2009,
          title = {The Variational Gaussian Approximation Revisited},
          author = {Opper, Manfred and Archambeau, Cedric},
          journal = {Neural Comput.},
          year = {2009},
          pages = {786--792},
      }
    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior. It turns out that the optimal
    posterior precision shares off-diagonal elements with the prior, so
    only the diagonal elements of the precision need be adjusted.
    The posterior approximation is
    .. math::
       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\mathbf K \\boldsymbol \\alpha, [\\mathbf K^{-1} + \\textrm{diag}(\\boldsymbol \\lambda))^2]^{-1})

    This approach has only 2ND parameters, rather than the N + N^2 of vgp,
    but the optimization is non-convex and in practice may cause difficulty.

    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=Zero(), num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects
        """
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
        self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                              transforms.positive)

    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                                  transforms.positive)
        return super(VGP_opper_archambeau, self).compile(session=session,
                                                         graph=graph,
                                                         optimizer=optimizer)

    def build_likelihood(self):
        """
        q_alpha, q_lambda are variational parameters, size N x R
        This method computes the variational lower bound on the likelihood,
        which is:
            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]
        with
            q(f) = N(f | K alpha + mean, [K^-1 + diag(square(lambda))]^-1) .
        """
        K = self.kern.K(self.X)
        K_alpha = tf.matmul(K, self.q_alpha)
        f_mean = K_alpha + self.mean_function(self.X)

        # compute the variance for each of the outputs
        I = tf.tile(tf.expand_dims(tf.eye(self.num_data, dtype=float_type), 0), [self.num_latent, 1, 1])
        A = I + tf.expand_dims(tf.transpose(self.q_lambda), 1) * \
            tf.expand_dims(tf.transpose(self.q_lambda), 2) * K
        L = tf.cholesky(A)
        Li = tf.matrix_triangular_solve(L, I)
        tmp = Li / tf.expand_dims(tf.transpose(self.q_lambda), 1)
        f_var = 1./tf.square(self.q_lambda) - tf.transpose(tf.reduce_sum(tf.square(tmp), 1))

        # some statistics about A are used in the KL
        A_logdet = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
        trAi = tf.reduce_sum(tf.square(Li))

        KL = 0.5 * (A_logdet + trAi - self.num_data * self.num_latent +
                    tf.reduce_sum(K_alpha*self.q_alpha))

        v_exp = self.likelihood.variational_expectations(f_mean, f_var, self.Y)
        return tf.reduce_sum(v_exp) - KL

    def build_predict(self, Xnew, full_cov=False):
        """
        The posterior variance of F is given by
            q(f) = N(f | K alpha + mean, [K^-1 + diag(lambda**2)]^-1)
        Here we project this to F*, the values of the GP at Xnew which is given
        by
           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda**-2)]^-1 K_{f*} )
        """

        # compute kernel things
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X)

        # predictive mean
        f_mean = tf.matmul(Kx, self.q_alpha, transpose_a=True) + self.mean_function(Xnew)

        # predictive var
        A = K + tf.matrix_diag(tf.transpose(1./tf.square(self.q_lambda)))
        L = tf.cholesky(A)
        Kx_tiled = tf.tile(tf.expand_dims(Kx, 0), [self.num_latent, 1, 1])
        LiKx = tf.matrix_triangular_solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kern.K(Xnew) - tf.matmul(LiKx, LiKx, transpose_a=True)
        else:
            f_var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(LiKx), 1)
        return f_mean, tf.transpose(f_var)
