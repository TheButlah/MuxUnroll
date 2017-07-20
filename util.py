from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import sys

from time import time
from bisect import bisect_left


def batch_norm(x, shape, axes, phase_train, decay=0.5, scope='BN'):
    """
    Batch normalization on tensors.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Note: The original author's code has been modified to generalize the spatial dimensions of the input tensor.
    Args:
        x:           Tensor to batch normalize
        shape:       Tuple, shape of input
        axes:        Tuple, axes along which mean and variance will be normalized
        phase_train: Boolean tf.Variable, true indicates training phase
        decay:       The decay rate for the exponential moving average
        scope:       String, variable scope
    Returns:
        normed:      Batch-normalized tensor
    """
    with tf.variable_scope(scope):
        try:
            remaining_axes = (shape[axis] for axis in axes)
        except IndexError:
            raise ValueError('The given axes are incompatible with the given shape.')

        beta = tf.Variable(tf.constant(0.0, shape=remaining_axes),
                           name='Beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=remaining_axes),
                            name='Gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, axes=axes, name='Moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def print_examples(model, batch, print_start_stop=True, print_timing=True):
    if print_start_stop: print("Applying model...")
    start_time = time()
    results = model.apply(batch)
    elapsed = time() - start_time
    if print_start_stop: print("Finished!")
    for x, y in zip(batch, results):
        print("\nInput:")
        print(x)
        print("Output:")
        print(y)
    if print_timing: print("\nInference Time: %f seconds" % elapsed)


def find_ge(a, x):
    """Find leftmost item greater than or equal to x"""
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    else:
        raise ValueError("The list does not have any values greater than or equal to:", x)


def eprint(*args, **kwargs):
    """Prints to stderr"""
    print(*args, file=sys.stderr, **kwargs)
