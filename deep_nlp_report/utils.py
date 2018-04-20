
import os
from pathlib import Path

import numpy as np

import tensorflow as tf


def this_scripts_path():
    return Path(os.path.dirname(os.path.realpath(__file__)))


def apply_mask(x, m):
    if isinstance(x, np.ndarray):
        return x[m]
    if isinstance(x, list):
        return [a for a, p in zip(x, m) if p]


def mask_all(*args, mask=None):
    if mask is None:
        raise ValueError('You have to provide a mask.')
    return [apply_mask(x, mask) for x in args]


def to_array(*args):
    return [np.array(x) for x in args]
        
        
def shuffle_all(xs):
    n = len(xs[0])
    # all same length
    assert np.all(len(x) == n for x in xs)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    return [x[idxs] for x in xs]
    
        
def batches(xs, batch_size):
    n = len(xs[0])
    assert np.all(len(x) == n for x in xs)
    xs = shuffle_all(xs)
    batch_start = 0
    while True:
        batch_end = batch_start + batch_size
        
        if  batch_end >= n:
            batch_start = 0
            batch_end = batch_start + batch_size
            
        yield [x[batch_start:batch_end] for x in xs]
        batch_start += batch_size
        
        
def tf_shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
        