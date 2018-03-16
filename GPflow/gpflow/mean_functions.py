# Copyright 2016 James Hensman, alexggmatthews, PabloLeon, Valentine Svensson
# Copyright 2018 KAIST under XAI Project supported by Ministry of Science and ICT, Korea
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


import tensorflow as tf
import numpy as np
from .param import Param, ParamList, Parameterized
from ._settings import settings
from layers import affine_relu_forward, affine_forward
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class MeanFunction(Parameterized):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """
    def __call__(self, X):
        raise NotImplementedError("Implement the __call__\
                                  method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)


class Zero(MeanFunction):
    def __call__(self, X, X_train=None): # modified
        return tf.zeros(tf.stack([tf.shape(X)[0], 1]), dtype=float_type)


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = np.ones((1, 1)) if A is None else A
        b = np.zeros(1) if b is None else b
        MeanFunction.__init__(self)
        self.A = Param(np.atleast_2d(A))
        self.b = Param(b)

    def __call__(self, X):
        return tf.matmul(X, self.A) + self.b


class Constant(MeanFunction):
    """
    y_i = c,,
    """
    def __init__(self, c=None):
        MeanFunction.__init__(self)
        c = np.zeros(1) if c is None else c
        self.c = Param(c)

    def __call__(self, X):
        shape = tf.stack([tf.shape(X)[0], 1])
        return tf.tile(tf.reshape(self.c, (1, -1)), shape)


class SwitchedMeanFunction(MeanFunction):
    """
    This class enables to use different (independent) mean_functions respective
    to the data 'label'.
    We assume the 'label' is stored in the extra column of X.
    """
    def __init__(self, meanfunction_list):
        MeanFunction.__init__(self)
        for m in meanfunction_list:
            assert isinstance(m, MeanFunction)
        self.meanfunction_list = ParamList(meanfunction_list)
        self.num_meanfunctions = len(self.meanfunction_list)

    def __call__(self, X):
        ind = tf.gather(tf.transpose(X), tf.shape(X)[1]-1)  # ind = X[:,-1]
        ind = tf.cast(ind, tf.int32)
        X = tf.transpose(tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1]-1)))  # X = X[:,:-1]

        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, self.num_meanfunctions)
        # apply the likelihood-function to each section of the data
        results = [m(x) for (x, m) in zip(x_list, self.meanfunction_list)]
        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_meanfunctions)
        return tf.dynamic_stitch(partitions, results)


class Additive(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X):
        return tf.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X):
        return tf.multiply(self.prod_1(X), self.prod_2(X))


class TwoLayerSigmoidMLP(MeanFunction):
    """
    (Custom) Two layer MLP with sigmoid activation function.
    """
    def __init__(self, W1=None, W2=None, b1=None, b2=None, input_dim=1, hidden_dim=10, output_dim=1, is_class=False):
        W1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W1 is None else W1
        W2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W2 is None else W2
        b1 = np.zeros(hidden_dim) if b1 is None else b1
        b2 = np.zeros(output_dim) if b2 is None else b2
        MeanFunction.__init__(self)
        self.W1 = Param(np.atleast_2d(W1))
        self.W2 = Param(np.atleast_2d(W2))
        self.b1 = Param(b1)
        self.b2 = Param(b2)

        self.is_class = is_class

    def __call__(self, X, X_train=None):
        if self.is_class:
            return tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(X, self.W1) + self.b1), self.W2) + self.b2)
        else:
            return tf.matmul(tf.sigmoid(tf.matmul(X, self.W1) + self.b1), self.W2) + self.b2


class MixtureExpertsMLP2(MeanFunction):
    """
    (Custom) Mixture of experts with MLP as basis function.
    Number of experts are 2.
    """
    def __init__(self, W1_1=None, W1_2=None, b1_1=None, b1_2=None, W2_1=None, W2_2=None, b2_1=None, b2_2=None, 
                 input_dim=1, hidden_dim=10, output_dim=1, is_class=False):
        # Expert 1 parameters
        W1_1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W1_1 is None else W1_1
        W1_2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W1_2 is None else W1_2
        b1_1 = np.zeros(hidden_dim) if b1_1 is None else b1_1
        b1_2 = np.zeros(output_dim) if b1_2 is None else b1_2
        # Expert 2 parameters
        W2_1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W2_1 is None else W2_1
        W2_2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W2_2 is None else W2_2
        b2_1 = np.zeros(hidden_dim) if b2_1 is None else b2_1
        b2_2 = np.zeros(output_dim) if b2_2 is None else b2_2
        # Gate softmax parameter
        W = 1e-1*np.random.randn(input_dim, 1) # expert numbers - 1
        MeanFunction.__init__(self)
        self.W1_1 = Param(np.atleast_2d(W1_1))
        self.W1_2 = Param(np.atleast_2d(W1_2))
        self.b1_1 = Param(b1_1)
        self.b1_2 = Param(b1_2)
        self.W2_1 = Param(np.atleast_2d(W2_1))
        self.W2_2 = Param(np.atleast_2d(W2_2))
        self.b2_1 = Param(b2_1)
        self.b2_2 = Param(b2_2)
        self.W = Param(np.atleast_2d(W))

    def __call__(self, X, X_train=None):
        e1 = tf.matmul(tf.sigmoid(tf.matmul(X, self.W1_1) + self.b1_1), self.W1_2) + self.b1_2
        e2 = tf.matmul(tf.sigmoid(tf.matmul(X, self.W2_1) + self.b2_1), self.W2_2) + self.b2_2
        g1 = tf.sigmoid(tf.matmul(X, self.W))
        g2 = 1 - g1
        return (e1*g1) + (e2*g2)


class MixtureExpertsMLP4(MeanFunction):
    """
    (Custom) Mixture of experts with MLP as basis function.
    Number of experts are 2.
    """
    def __init__(self, W1_1=None, W1_2=None, b1_1=None, b1_2=None, W2_1=None, W2_2=None, b2_1=None, b2_2=None, 
                 W3_1=None, W3_2=None, b3_1=None, b3_2=None, W4_1=None, W4_2=None, b4_1=None, b4_2=None, 
                 input_dim=1, hidden_dim=10, output_dim=1, is_class=False):
        # Expert 1 parameters
        W1_1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W1_1 is None else W1_1
        W1_2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W1_2 is None else W1_2
        b1_1 = np.zeros(hidden_dim) if b1_1 is None else b1_1
        b1_2 = np.zeros(output_dim) if b1_2 is None else b1_2
        # Expert 2 parameters
        W2_1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W2_1 is None else W2_1
        W2_2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W2_2 is None else W2_2
        b2_1 = np.zeros(hidden_dim) if b2_1 is None else b2_1
        b2_2 = np.zeros(output_dim) if b2_2 is None else b2_2
        # Expert 3 parameters
        W3_1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W3_1 is None else W3_1
        W3_2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W3_2 is None else W3_2
        b3_1 = np.zeros(hidden_dim) if b3_1 is None else b3_1
        b3_2 = np.zeros(output_dim) if b3_2 is None else b3_2
        # Expert 4 parameters
        W4_1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W4_1 is None else W4_1
        W4_2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W4_2 is None else W4_2
        b4_1 = np.zeros(hidden_dim) if b4_1 is None else b4_1
        b4_2 = np.zeros(output_dim) if b4_2 is None else b4_2
        # Gate softmax parameter
        W = 1e-1*np.random.randn(input_dim, 4) # expert numbers
        MeanFunction.__init__(self)
        self.W1_1 = Param(np.atleast_2d(W1_1))
        self.W1_2 = Param(np.atleast_2d(W1_2))
        self.b1_1 = Param(b1_1)
        self.b1_2 = Param(b1_2)
        self.W2_1 = Param(np.atleast_2d(W2_1))
        self.W2_2 = Param(np.atleast_2d(W2_2))
        self.b2_1 = Param(b2_1)
        self.b2_2 = Param(b2_2)
        self.W3_1 = Param(np.atleast_2d(W3_1))
        self.W3_2 = Param(np.atleast_2d(W3_2))
        self.b3_1 = Param(b3_1)
        self.b3_2 = Param(b3_2)
        self.W4_1 = Param(np.atleast_2d(W4_1))
        self.W4_2 = Param(np.atleast_2d(W4_2))
        self.b4_1 = Param(b4_1)
        self.b4_2 = Param(b4_2)
        self.W = Param(np.atleast_2d(W))

    def __call__(self, X, X_train=None):
        e1 = tf.matmul(tf.sigmoid(tf.matmul(X, self.W1_1) + self.b1_1), self.W1_2) + self.b1_2
        e2 = tf.matmul(tf.sigmoid(tf.matmul(X, self.W2_1) + self.b2_1), self.W2_2) + self.b2_2
        e3 = tf.matmul(tf.sigmoid(tf.matmul(X, self.W3_1) + self.b3_1), self.W3_2) + self.b3_2
        e4 = tf.matmul(tf.sigmoid(tf.matmul(X, self.W4_1) + self.b4_1), self.W4_2) + self.b4_2
        g = tf.nn.softmax(tf.matmul(X, self.W))
        g1 = g[:,0:1]
        g2 = g[:,1:2]
        g3 = g[:,2:3]
        g4 = g[:,3:4]
        return (e1*g1) + (e2*g2) + (e3*g3) + (e4*g4)


class TwoLayerReLUMLP(MeanFunction):
    """
    (Custom) Two layer MLP with ReLU activation function.
    """
    def __init__(self, W1=None, W2=None, b1=None, b2=None, input_dim=1, hidden_dim=10, output_dim=1, is_class=False):
        W1 = 1e-1*np.random.randn(input_dim, hidden_dim) if W1 is None else W1
        W2 = 1e-1*np.random.randn(hidden_dim, output_dim) if W2 is None else W2
        b1 = np.zeros(hidden_dim) if b1 is None else b1
        b2 = np.zeros(output_dim) if b2 is None else b2
        MeanFunction.__init__(self)
        self.W1 = Param(np.atleast_2d(W1))
        self.W2 = Param(np.atleast_2d(W2))
        self.b1 = Param(b1)
        self.b2 = Param(b2)

        self.is_class = is_class

    def __call__(self, X, X_train=None):
        if self.is_class:
            return tf.sigmoid(tf.matmul(tf.maximum(tf.matmul(X, self.W1) + self.b1, 0), self.W2) + self.b2)
        else:
            return tf.matmul(tf.maximum(tf.matmul(X, self.W1) + self.b1, 0), self.W2) + self.b2


class RNN_OneLayer(MeanFunction):
    """
    (Custom) Vanilla RNN with one hidden layer.
    """
    def __init__(self, Wemb=None, W=None, Wout=None, bemb=None, b=None, bout=None, \
                 input_dim=1, hidden_dim=10, output_dim=1, length=None):
        Wemb = 1e-1*np.random.randn(input_dim, hidden_dim) if Wemb is None else Wemb
        W = 1e-1*np.random.randn(2*hidden_dim, hidden_dim) if W is None else W
        Wout = 1e-1*np.random.randn(hidden_dim, output_dim) if Wout is None else Wout
        b = np.zeros(hidden_dim) if b is None else b
        bout = np.zeros(output_dim) if bout is None else bout
        MeanFunction.__init__(self)
        self.Wemb = Param(np.atleast_2d(Wemb))
        self.W = Param(np.atleast_2d(W))
        self.Wout = Param(np.atleast_2d(Wout))
        self.b = Param(b)
        self.bout = Param(bout)
        
        self.length = length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # dummy
        W2 = np.zeros((2*hidden_dim, hidden_dim))
        self.W2 = Param(np.atleast_2d(W2))
       
        # TODO Also, have to implement evaluating training points

    def __call__(self, X, is_predict=False, X_train=None):
        if is_predict:
            length = 1
            H_pred = tf.concat([X_train, X], axis=0)
            out_pred = tf.matmul(H_pred, self.Wout) + self.bout
            return tf.reshape(out_pred[-length:, :], [length, -1])
        else:    
            H = X
            return tf.matmul(H, self.Wout) + self.bout


class RNN_TwoLayer(MeanFunction):
    """
    (Custom) Vanilla RNN with two hidden layer.
    """
    def __init__(self, Wemb=None, W=None, W2=None, Wout=None, bemb=None, b=None, b2=None, bout=None, \
                 input_dim=1, hidden_dim=10, output_dim=1, length=None):
        Wemb = 1e-1*np.random.randn(input_dim, hidden_dim) if Wemb is None else Wemb
        W = 1e-1*np.random.randn(2*hidden_dim, hidden_dim) if W is None else W
        W2 = 1e-1*np.random.randn(2*hidden_dim, hidden_dim) if W2 is None else W2
        Wout = 1e-1*np.random.randn(hidden_dim, output_dim) if Wout is None else Wout
        b = np.zeros(hidden_dim) if b is None else b
        b2 = np.zeros(hidden_dim) if b2 is None else b2
        bout = np.zeros(output_dim) if bout is None else bout
        MeanFunction.__init__(self)
        self.Wemb = Param(np.atleast_2d(Wemb))
        self.W = Param(np.atleast_2d(W))
        self.W2 = Param(np.atleast_2d(W2))
        self.Wout = Param(np.atleast_2d(Wout))
        self.b = Param(b)
        self.b2 = Param(b2)
        self.bout = Param(bout)
        
        self.length = length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
       
        # TODO Also, have to implement evaluating training points

    def __call__(self, X, is_predict=False, X_train=None):
        if is_predict:
            length = 1
            H_embedded_pred = tf.matmul(tf.concat([X_train, X], axis=0), self.W2[0:self.hidden_dim, :])

            H2_pred = []
            for i in range(self.length+length):
                if i == 0:
                    H2_pred.append(tf.tanh(H_embedded_pred[i,:]))
                else:
                    toAdd_pred = tf.matmul(tf.reshape(H2_pred[i-1], [1, -1]), \
                                           self.W2[self.hidden_dim:2*self.hidden_dim, :]) + self.b2
                    H2_pred.append(tf.tanh(H_embedded_pred[i,:] + toAdd_pred[0, :]))
            H2_pred = tf.stack(H2_pred, axis=0)

            out_pred = tf.matmul(H2_pred, self.Wout) + self.bout
            return tf.reshape(out_pred[-length:, :], [length, -1])
        else:    
            H_embedded = tf.matmul(X, self.W2[0:self.hidden_dim, :])

            H2 = []
            for i in range(self.length):
                if i == 0:
                    H2.append(tf.tanh(H_embedded[i,:]))
                else:
                    toAdd = tf.matmul(tf.reshape(H2[i-1], [1, -1]), \
                                      self.W2[self.hidden_dim:2*self.hidden_dim, :]) + self.b2
                    H2.append(tf.tanh(H_embedded[i,:] + toAdd[0, :]))
            H2 = tf.stack(H2, axis=0)

            return tf.matmul(H2, self.Wout) + self.bout


class RNN_OneLayer_DKGP(MeanFunction):
    """
    (Custom) Vanilla RNN with one hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, Wout=None, bout=None):
        Wout = 1e-1*np.random.randn(input_dim, output_dim) if Wout is None else Wout
        bout = np.zeros(output_dim) if bout is None else bout
        
        MeanFunction.__init__(self)
        self.Wout = Param(np.atleast_2d(Wout))
        self.bout = Param(bout)
        
    def __call__(self, X, X_train=None):
        return tf.matmul(X, self.Wout) + self.bout


class RNN_TwoLayer_DKGP(MeanFunction):
    """
    (Custom) Vanilla RNN with two hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, W2=None, Wout=None, b2=None, bout=None):
        W2 = 1e-1*np.random.randn(input_dim+hidden_dim, hidden_dim) if W2 is None else W2
        Wout = 1e-1*np.random.randn(hidden_dim, output_dim) if Wout is None else Wout
        b2 = np.zeros(hidden_dim) if b2 is None else b2
        bout = np.zeros(output_dim) if bout is None else bout
        
        MeanFunction.__init__(self)
        self.W2 = Param(np.atleast_2d(W2))
        self.Wout = Param(np.atleast_2d(Wout))
        self.b2 = Param(b2)
        self.bout = Param(bout)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
       
    def __call__(self, X, X_train=None):
        length = tf.shape(X)[0]

        if X_train == None:
            _X = X
            _length = length
        else:
            _X = tf.cond(tf.equal(length, 1), lambda: tf.concat([X_train, X], axis=0), lambda: X)
            _length = tf.cond(tf.equal(length, 1), lambda: tf.shape(_X)[0], lambda: length)

        Xemb = tf.matmul(_X, self.W2[0:self.input_dim, :])

        i = tf.constant(1)
        H = tf.reshape(tf.tanh(Xemb[0, :]), [1,-1])
        def cond(i, Xemb, H, _length):
            return i < _length
        def body(i, Xemb, H, _length):
            toAdd = tf.matmul(tf.reshape(H[i-1, :], [1,-1]), self.W2[self.input_dim:self.input_dim+self.hidden_dim, :]) + self.b2
            toConcat = tf.tanh(tf.reshape(Xemb[i, :], [1,-1]) + toAdd)
            return [tf.add(i, 1), Xemb, tf.concat([H, toConcat], axis=0), _length]
        loop_vars = [i, Xemb, H, _length]
        shape_invariants = [i.get_shape(), Xemb.get_shape(), tf.TensorShape([None, self.hidden_dim]), _length.get_shape()]
        _, _, H, _ = tf.while_loop(cond, body, loop_vars=loop_vars, shape_invariants=shape_invariants)

        if X_train == None:
            pass
        else:
            H = tf.cond(tf.equal(length, 1), lambda: tf.reshape(H[-1, :], [1,-1]), lambda: H)

        return tf.matmul(H, self.Wout) + self.bout
