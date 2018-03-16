import numpy as np
import tensorflow as tf


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
  
    Inputs:
        x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        w: A numpy array of weights, of shape (D, M)
    b: A numpy array of biases, of shape (M,)
    
    Returns:
        out: output, of shape (N, M)
        cache: (x, w, b)
    """
#    X = tf.reshape(x, [x.get_shape().as_list()[0], -1]) # TODO generalize
    out = tf.matmul(x, w) + b
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    
    Inputs:
        dout: Upstream derivative, of shape (N, M)
        cache: Tuple of:
            x: Input data, of shape (N, d_1, ... d_k)
            w: Weights, of shape (D, M)
    
    Returns:
        dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        dw: Gradient with respect to w, of shape (D, M)
        db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    X = tf.reshape(x, [x.shape[0], -1]) # (N, M)
    dx = tf.reshape(tf.matmul(dout, tf.transpose(w)), tf.shape(x))
    dw, db = tf.matmul(tf.transpose(X), dout), tf.reduce_sum(dout, 0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    
    Inputs:
        x: Inputs, of any shape

    Returns a tuple of:
        out: Output, of the same shape as x
        cache: x
    """
    out = tf.maximum(x, 0)
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Inputs:
        dout: Upstream derivatives, of any shape
        cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.array(dout, copy=True)
    dx[x<=0] = 0

    return dx


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
        x: Input to the affine layer
        w, b: Weights for the affine layer

    Returns:
        out: Output from the ReLU
        cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)

    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db


def dL_do(o, y, K_inv):
    """
    Compute gradient of last layer of MLP.
    """
    return np.matmul(K_inv, y - o)

