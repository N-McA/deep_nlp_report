

import keras.backend as K
import keras


class GlobalAveragePooling1DMasked(keras.layers.GlobalAveragePooling1D):
    def __init__(self):
        super().__init__()
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
