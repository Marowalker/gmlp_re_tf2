from tensorflow.keras.layers import Layer, LayerNormalization
import tensorflow as tf


class PreNorm(Layer):
    def __init__(self, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)
        self.fn = fn

    def build(self, input_shape):
        self.norm = LayerNormalization(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        return self.fn(self.norm(inputs))
