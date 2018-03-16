from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .param import Param, Parameterized


class FeatureRepresentation(Parameterized):
    """
    The base class of feature representation for DKGP such as MLP, RNN.
    """
    def __call__(self, X):
        raise NotImplementedError("Implement the __call__ method for this feature representation class")


class Identity(FeatureRepresentation):
    """
    Do not use feature learning, which means the model is going to be GPR.
    """
    def __call__(self, X, X_train=None):
        return X


class MLP(FeatureRepresentation):
    """
    Feature representation by One hidden layer MLP.
    """
    def __init__(self, input_dim, hidden_dim, params=None):
        """
        Initialize parameters in MLP.
        
        args:
            input_dim - input dimension of the data
            hidden_dim - hidden dimension of the features
            params - already been defined parameters as dict, else None
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if params != None:
            self.check_init_params(params)
            W1, W2, b1, b2 = params['W1'], params['W2'], params['b1'], params['b2']
        else:
            W1 = 1e-1*np.random.randn(input_dim, hidden_dim)
            b1 = np.zeros(hidden_dim)
            W2 = 1e-1*np.random.randn(hidden_dim, hidden_dim)
            b2 = np.zeros(hidden_dim)

        FeatureRepresentation.__init__(self)
        self.W1 = Param(np.atleast_2d(W1))
        self.b1 = Param(b1)
        self.W2 = Param(np.atleast_2d(W2))
        self.b2 = Param(b2)

    def __call__(self, X, X_train=None):
        """
        Compute feature representation of the data.
        """
        return tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(X, self.W1) + self.b1), self.W2) + self.b2)

    def check_init_params(self, params):
        assert len(params) == 4, 'A number of parameters is not correct'
        assert 'W1' in params.keys(), 'W1 does not exist'
        assert 'W2' in params.keys(), 'W2 does not exist'
        assert 'b1' in params.keys(), 'b1 does not exist'
        assert 'b2' in params.keys(), 'b2 does not exist'


class RNN(FeatureRepresentation):
    """
    Feature representation by One hidden layer RNN
    """
    def __init__(self, input_dim, hidden_dim, params=None):
        """
        Initialize parameters in RNN.
        
        args:
            input_dim - input dimension of the data
            hidden_dim - hidden dimension of the features
            params - already been defined parameters as dict, else None
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if params != None:
            self.check_init_params(params)
            Wemb, W, b = params['Wemb'], params['W'], params['b']
        else:
            Wemb = 1e-1*np.random.randn(input_dim, hidden_dim)
            W = 1e-1*np.random.randn(2*hidden_dim, hidden_dim)
            b = np.zeros(hidden_dim)

        FeatureRepresentation.__init__(self)
        self.Wemb = Param(np.atleast_2d(Wemb))
        self.W = Param(np.atleast_2d(W))
        self.b = Param(b)

    def __call__(self, X, X_train=None):
        """
        Compute feature representation of the data.
        Calling must be defined separately into cases of when training and testing
        """
        length = tf.shape(X)[0]

        if X_train == None:
            _X = X
            _length = length
        else:
            _X = tf.cond(tf.equal(length, 1), lambda: tf.concat([X_train, X], axis=0), lambda: X)
            _length = tf.cond(tf.equal(length, 1), lambda: tf.shape(_X)[0], lambda: length)

        Xemb = tf.matmul(tf.matmul(_X, self.Wemb), self.W[0:self.hidden_dim, :])

        i = tf.constant(1)
        H = tf.reshape(tf.tanh(Xemb[0, :]), [1,-1])
        def cond(i, Xemb, H, _length):
            return i < _length
        def body(i, Xemb, H, _length):
            toAdd = tf.matmul(tf.reshape(H[i-1, :], [1,-1]), self.W[self.hidden_dim:2*self.hidden_dim, :]) + self.b
            toConcat = tf.tanh(tf.reshape(Xemb[i, :], [1,-1]) + toAdd)
            return [tf.add(i, 1), Xemb, tf.concat([H, toConcat], axis=0), _length]
        loop_vars = [i, Xemb, H, _length]
        shape_invariants = [i.get_shape(), Xemb.get_shape(), tf.TensorShape([None, self.hidden_dim]), _length.get_shape()]
        _, _, H, _ = tf.while_loop(cond, body, loop_vars=loop_vars, shape_invariants=shape_invariants)

        if X_train == None:
            pass
        else:
            H = tf.cond(tf.equal(length, 1), lambda: tf.reshape(H[-1, :], [1,-1]), lambda: H)

        return H

    def check_init_params(self, params):
        assert len(params) == 3, 'A number of parameters is not correct'
        assert 'Wemb' in params.keys(), 'Wemb does not exist'
        assert 'W' in params.keys(), 'W does not exist'
        assert 'b' in params.keys(), 'b does not exist'
