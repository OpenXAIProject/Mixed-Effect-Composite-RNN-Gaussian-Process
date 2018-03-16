# Written by Ingyo Chung, KAIST, KOREA


from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .model import GPModel
from .densities import multivariate_normal
from .mean_functions import Zero, RNN_OneLayer_DKGP, RNN_TwoLayer_DKGP, TwoLayerSigmoidMLP
from . import likelihoods
from .param import DataHolder, Param, Parameterized
from .minibatch import MinibatchData
from ._settings import settings
from . import transforms, kullback_leiblers, conditionals
from .conditionals import conditional
from .kullback_leiblers import gauss_kl
from .feature_representations import MLP, RNN, Identity 
float_type = settings.dtypes.float_type


class DKGPR(GPModel):
    """
    Deep Kernel Gaussian Process Regression.

    Feature learning models such as MLP will be attached to the inputs of GP.
    
    """
    def __init__(self, X, Y, kern, mean_function=None, feature_representation=None, name='name'):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        self.feature_representation = feature_representation or Identity()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, name)
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        H = self.feature_representation(self.X)
        K = self.kern.K(H) + tf.eye(tf.shape(H)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(H)

        return multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        H = self.feature_representation(self.X)
        Hnew = self.feature_representation(Xnew)
        Kx = self.kern.K(H, Hnew)
        K = self.kern.K(H) + tf.eye(tf.shape(H)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(H))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Hnew)
        if full_cov:
            fvar = self.kern.K(Hnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Hnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar


class DKVGP(GPModel):
    """
    Deep Kernel Gaussain Process approximated by SVGP for classification.

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
                 mean_function=None, feature_representation=None, num_latent=None, reg=0.0):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        """
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        self.num_data = X.shape[0]
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_latent = num_latent or Y.shape[1]
        self.feature_representation = feature_representation or Identity()
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

        return super(DKVGP, self).compile(session=session,
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

        # Feature representation of hidden states of RNN inspired by DKL
        H = self.feature_representation(self.X)

        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
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

        # L2 regularization
        reg = 0
        if isinstance(self.feature_representation, MLP):
            reg += tf.reduce_sum(tf.square(self.feature_representation.W1)) + \
                   tf.reduce_sum(tf.square(self.feature_representation.W2))
        elif isinstance(self.feature_representation, RNN):
            reg += tf.reduce_sum(tf.square(self.feature_representation.Wemb)) + \
                   tf.reduce_sum(tf.square(self.feature_representation.W))
        
        if isinstance(self.mean_function, RNN_OneLayer_DKGP):
            reg += tf.reduce_sum(tf.square(self.mean_function.Wout))
        elif isinstance(self.mean_function, RNN_TwoLayer_DKGP):
            reg += tf.reduce_sum(tf.square(self.mean_function.W2)) + \
                   tf.reduce_sum(tf.square(self.mean_function.Wout))
        elif isinstance(self.mean_function, TwoLayerSigmoidMLP):
            reg += tf.reduce_sum(tf.square(self.mean_function.W1)) + \
                   tf.reduce_sum(tf.square(self.mean_function.W2))

        return tf.reduce_sum(var_exp) - KL - (.5 * self.reg * reg)

    def build_predict(self, Xnew, full_cov=False):
        H = self.feature_representation(self.X)
        Hnew = self.feature_representation(Xnew, X_train=self.X)
        mu, var = conditional(Hnew, H, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Hnew, X_train=H), var


class DKSVGP(GPModel):
    """
    Deep kernel GP approximated by SVGP.

    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """
    def __init__(self, X, Y, kern, likelihood, Z, mean_function=None, feature_representation=None,
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None, reg=0.0):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        """
        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Param(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0]
        self.feature_representation = feature_representation or Identity()
        self.reg = reg

        # init variational parameters
        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent)))
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_inducing, self.num_latent)),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = Param(q_sqrt, transforms.LowerTriangular(self.num_inducing, self.num_latent))

    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            HZ = self.feature_representation(self.Z)
            K = self.kern.K(HZ) + tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self.build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        # L2 regularization
        reg = 0
        if isinstance(self.feature_representation, MLP):
            reg += tf.reduce_sum(tf.square(self.feature_representation.W1)) + \
                   tf.reduce_sum(tf.square(self.feature_representation.W2))
        elif isinstance(self.feature_representation, RNN):
            reg += tf.reduce_sum(tf.square(self.feature_representation.Wemb)) + \
                   tf.reduce_sum(tf.square(self.feature_representation.W))
        
        if isinstance(self.mean_function, RNN_OneLayer_DKGP):
            reg += tf.reduce_sum(tf.square(self.mean_function.Wout))
        elif isinstance(self.mean_function, RNN_TwoLayer_DKGP):
            reg += tf.reduce_sum(tf.square(self.mean_function.W2)) + \
                   tf.reduce_sum(tf.square(self.mean_function.Wout))
        elif isinstance(self.mean_function, TwoLayerSigmoidMLP):
            reg += tf.reduce_sum(tf.square(self.mean_function.W1)) + \
                   tf.reduce_sum(tf.square(self.mean_function.W2))

        return tf.reduce_sum(var_exp) * scale - KL - (.5 * self.reg * reg)

    def build_predict(self, Xnew, full_cov=False):
        Hnew = self.feature_representation(Xnew)
        HZ = self.feature_representation(self.Z)
        mu, var = conditionals.conditional(Hnew, HZ, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Hnew), var
