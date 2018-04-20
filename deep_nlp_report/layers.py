

import keras.backend as K
import keras

import tensorflow as tf
from deep_nlp_report.utils import tf_shape_list
import math


class GlobalAveragePooling1DMasked(keras.layers.GlobalAveragePooling1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, x, mask=None):
        if mask != None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            return K.sum(x * mask, axis=-2) / K.sum(mask, axis=-2)
        else:
            return super().call(x)
        
    def compute_mask(self, inputs, mask=None):
        return None
    

class GlobalSumPooling1DMasked(keras.layers.GlobalAveragePooling1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, x, mask=None):
        if mask != None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            return K.sum(x * mask, axis=-2)
        else:
            return super().call(x)
        
    def compute_mask(self, inputs, mask=None):
        return None
    
    
def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """https://github.com/tensorflow/tensor2tensor/blob/120315cbe35f468876512b790dc77d792d4db72c/tensor2tensor/layers/common_attention.py#L504
    Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase in one of the positional dimensions.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(a+b) and cos(a+b) can be
    experessed in terms of b, sin(a) and cos(a).
    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image
    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    num_dims = len(x.get_shape().as_list()) - 2
    channels = tf_shape_list(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in range(num_dims):
        length = tf_shape_list(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x

    
class AddTimingSignalLayer(keras.engine.Layer):
    
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4):
        super().__init__()
        self.supports_masking = True
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        
    def call(self, x, mask=None):
        # We can ignore the masks.
        return add_timing_signal_nd(x, self.min_timescale, self.max_timescale)
        
    def compute_mask(self, inputs, mask=None):
        return mask