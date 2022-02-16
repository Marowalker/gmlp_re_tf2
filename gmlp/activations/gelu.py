from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from math import pi


class GELU(Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return (inputs / 2.) * (1 + tf.math.tanh(
            tf.math.sqrt(2. / tf.constant(pi)) * (inputs + 0.044715 * tf.math.pow(inputs, 3))
        ))
