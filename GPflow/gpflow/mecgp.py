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
from .param import Param, DataHolder, AutoFlow
from .model import GPModel_MECGP
from .mean_functions import Zero, TwoLayerSigmoidMLP, TwoLayerReLUMLP, MixtureExpertsMLP2, MixtureExpertsMLP4
from ._settings import settings
from . import transforms
from .conditionals import conditional
from .kullback_leiblers import gauss_kl

from sklearn.metrics import roc_auc_score

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type


class MECGP(GPModel_MECGP):
    """
    TODO: Description of MECGP
    """
    def __init__(self, X, Y, kern, likelihood, mean_function, 
                 feature_representation=None, num_latent=None, reg=0.0, isTest=False):
        """
        X is a dictionary of multi-task data, {P:Np x Dp}.
        Y is a dictionary of multu-task data, {P:Np x Rp}.
        kern is a dictionary of appropriate GPflow kernels of size P.
        likelihood is an appropriate GPflow object.
        mean_function is an appropriate GPflow object representing global mean function in MECGP.
        reg is a regularization constant for mean function.
        """ 
        self.X_train = {p: np.copy(Xp) for p, Xp in X.items()}
        self.Y_train = {p: np.copy(Yp) for p, Yp in Y.items()}
        
        self.p_id = [p for p in X.keys()]
        if isTest:
            self.size = {p: 1 for p, v in X.items()}
        else:
            self.size = {p: v.shape[0] for p, v in X.items()}
        X = {p: DataHolder(Xp, on_shape_change='recompile') for p, Xp in X.items()}
        Y = {p: DataHolder(Yp, on_shape_change='recompile') for p, Yp in Y.items()}
        GPModel_MECGP.__init__(self, X, Y, kern, likelihood, mean_function, feature_representation, 
                               self.p_id, self.size)
        self.num_data = {p: Xp.shape[0] for p, Xp in X.items()}
        self.num_latent = 1 # just for our experimental setting.
        self.reg = reg

        # variational parameters.
        self.q_mu = {p: Param(np.zeros((self.num_data[p], self.num_latent))) for p in self.p_id}
        self.q_sqrt = {}
        for p in self.p_id:
            q_sqrt = np.array([np.eye(self.num_data[p]) for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt[p] = Param(q_sqrt, transforms.LowerTriangular(self.num_data[p], self.num_latent))

    def compile(self, session=None, graph=None, optimizer=None):
        """
        TODO: For now check nothing.
        """
        return super(MECGP, self).compile(session=session, graph=graph, optimizer=optimizer)

    def build_likelihood(self):
        """
        Build likelihood for MECGP at once, aggregating all likelihoods of all GPs.
        """
        likelihood = 0

        for p in self.p_id:
            H = self.feature_representation(self.X[p])
            KL = gauss_kl(self.q_mu[p], self.q_sqrt[p])
            K = self.kern[p].K(H) + tf.eye(self.num_data[p], dtype=float_type)*settings.numerics.jitter_level
            L = tf.cholesky(K)
            fmean = tf.matmul(L, self.q_mu[p]) + self.mean_function(H)  # NN,ND->ND
            q_sqrt_dnn = tf.matrix_band_part(tf.transpose(self.q_sqrt[p], [2, 0, 1]), -1, 0)  # D x N x N
            L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent, 1, 1]))
            LTA = tf.matmul(L_tiled, q_sqrt_dnn)  # D x N x N
            fvar = tf.reduce_sum(tf.square(LTA), 2)
            fvar = tf.transpose(fvar)
            var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y[p])
            
            #TODO for now exclude regularization.

            likelihood += tf.reduce_sum(var_exp) - KL

        return likelihood

    def build_predict(self, Xnew, full_cov=False):
        """
        Predict testing points given vertically concatnated (axis=0) Xnew for all patients.
        """
        mu_toStack = []
        var_toStack = []
        size = 0
        for p in self.p_id:
            H = self.feature_representation(self.X[p])
            Hnew = self.feature_representation(Xnew[size:size+self.size[p],:], X_train=self.X[p])
            mu, var = conditional(Hnew, H, self.kern[p], self.q_mu[p],
                                  q_sqrt=self.q_sqrt[p], full_cov=full_cov, whiten=True)
            mu_toStack.append(mu + self.mean_function(Hnew, X_train=H))
            var_toStack.append(var)
            size += self.size[p]
        return tf.concat(mu_toStack, axis=0), tf.concat(var_toStack, axis=0)

    def optimize(self, method='CG', tol=None, callback=None, maxiter=1000, **kw):
        """
        Override 'optimize' method in the class 'Model' to adapt our model's optimization scheme.
        Since fixing parameters require recompilation, which means reset tensorflow graph,
        alternative optimization should be performed in run-time.
        """
        iters = 0
        conv_flag = False
        cond = lambda i,c: (i < maxiter) and (not c) #TODO convergence criterion
        while(cond(iters, conv_flag)):
            if type(method) is str:
                res = self._optimize_np(method, tol, callback, 1, **kw)
            else:
                res = self._optimize_tf(method, callback, 1, **kw)
            iters += 1
        return res

    def fix_parameters(self, part=None):
        if part == 'GPs':
            #unfix all parameters in mean function and feature_representation and fix all parameters in GPs.
            self.mean_function.fixed = False
            self.feature_representation.fixed = False
            for p in self.p_id:
                self.kern[p].fixed = True
                self.q_mu[p].fixed = True
                self.q_sqrt[p].fixed = True
        elif part == 'MF':
            #unfix all parameters in GPs and fix all parameters in mean function and feature_representation.
            for p in self.p_id:
                self.kern[p].fixed = False
                self.q_mu[p].fixed = False
                self.q_sqrt[p].fixed = False
            self.mean_function.fixed = True
            self.feature_representation.fixed = True
        else:
            raise ValueError('Fix right parameters')

    def sync_parameters(self, m, part=None):
        if part == 'GPs':
            # given MF, FR fixed model 'm', sync GPs of 'm' for 'self'.
            for p in self.p_id:
                self.kern[p].set_parameter_dict(m.kern[p].get_parameter_dict())
                self.q_mu[p]._array = m.q_mu[p].value
                self.q_sqrt[p]._array = m.q_sqrt[p].value
        elif part == 'MF':
            # given GPs fixed model 'm', sync MF, MR of 'm' for 'self'.
            self.mean_function.set_parameter_dict(m.mean_function.get_parameter_dict())
            self.feature_representation.set_parameter_dict(m.feature_representation.get_parameter_dict())
        else:
            raise ValueError('Sync right parameters')

    def print_intermediate_results(self, res, step, phase, Xnew=None, Ynew=None):
        if Xnew == None or Ynew == None:
            Xnew, Ynew = self.X_train, self.Y_train
        likelihood_train = res['fun']
        auc_train = self.compute_auc(Xnew, Ynew)

        print('%3d steps | Likelihood %.3f | %s AUC %.3f' %(step, res['fun'], phase, auc_train))
        return auc_train

    def compute_auc(self, X, Y):
        Y_hat = np.empty((0, self.num_latent))
        Y_true = np.empty((0, self.num_latent))

        _Y_hat, _Y_std = self.predict_y(X)

        for p in self.p_id:
            Y_hat = np.append(Y_hat, _Y_hat[p], axis=0)
            Y_true = np.append(Y_true, Y[p], axis=0)

        return roc_auc_score(Y_true, Y_hat)
