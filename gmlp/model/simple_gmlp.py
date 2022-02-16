import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding
from data_utils import *

import constants


class gMLPLayer(Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super(gMLPLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.norm = LayerNormalization(epsilon=1e-6)
        self.proj_in = Sequential([
            Dense(units=input_shape[-1] * 2, activation='gelu'),
            Dropout(rate=self.dropout_rate),
        ])
        self.sgu = SpatialGatingUnit(input_shape[-2])
        self.proj_out = Dense(input_shape[-1])

    def call(self, inputs, **kwargs):
        shortcut = self.norm(inputs)
        x = self.proj_in(shortcut)
        x = self.sgu(x)
        x = self.proj_out(x)

        return x + shortcut


class SpatialGatingUnit(Layer):
    def __init__(self, dim_in, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.dim_in = dim_in

    def build(self, input_shape):
        self.norm = LayerNormalization(epsilon=1e-6)
        self.proj = Dense(units=self.dim_in, bias_initializer="Ones")

    def call(self, inputs, **kwargs):
        u, v = tf.split(inputs, 2, axis=-1)

        v = self.norm(v)

        v = tf.linalg.matrix_transpose(v)
        v = self.proj(v)
        v = tf.linalg.matrix_transpose(v)

        return u * v


class gMLP(tf.keras.Model):
    def __init__(self, depth, dropout=0.1):
        super(gMLP, self).__init__()

        layers = []

        for i in range(depth):
            layers.append(gMLPLayer)

        self.models = tf.keras.models.Sequential(layers)

    def call(self, inputs, training=None, mask=None):
        return self.models(inputs, training=training)


class SimplegMLP(tf.keras.Model):
    def __init__(self, depth, embeddings, triples, wordnet):
        super(SimplegMLP, self).__init__()
        self.depth = depth
        self.embedding_dim = constants.INPUT_W2V_DIM
        self.seq_len = constants.MAX_LENGTH
        self.embeddings = embeddings
        self.triples = triples
        self.wordnet = wordnet

        self.num_of_words = countVocab()
        self.num_of_siblings = self.num_of_words
        self.num_of_poses = countNumPos()
        self.num_of_depend = countNumRelation()
        self.num_of_synsets = countNumSynset()

        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

        embedding_rel = tf.Variable(self.initializer(shape=[self.num_of_depend, self.embedding_dim],
                                                     dtype=tf.float32), name="re_lut", trainable=True)
        word_embedding = tf.concat([embeddings, embedding_rel], axis=0)
        self.to_embed = Embedding(self.num_of_words + self.num_of_depend + 1, self.embedding_dim,
                                  input_length=self.seq_len, weights=[word_embedding])

        self.pos_embed = Embedding(self.num_of_poses + self.num_of_depend + 1, 6, input_length=self.seq_len)

        synset_rel_emb = tf.Variable(self.initializer(shape=[self.num_of_depend, 17]),
                                     dtype=tf.float32, trainable=True)
        synset_emb = tf.concat([wordnet, synset_rel_emb], axis=0)
        self.synset_embed = Embedding(self.num_of_synsets + self.num_of_depend + 1, 17, input_length=self.seq_len,
                                      weights=[synset_emb])

        self.gmlp = gMLP(depth=depth)

    def call(self, inputs, training=None, mask=None):
        x, x_pos, x_synset, x_posi1, x_posi2, x_triple, x_sibling = inputs

        x = tf.cast(x, dtype='int64')
        x_pos = tf.cast(x_pos, dtype='int64')
        x_synset = tf.cast(x_synset, dtype='int64')
        x_posi1 = tf.cast(x_posi1, dtype='int64')
        x_posi2 = tf.cast(x_posi2, dtype='int64')
        x_triple = tf.cast(x_triple, dtype='int64')
        x_sibling = tf.cast(x_sibling, dtype='int64')

        x = self.to_embed(x)
        x_pos = self.pos_embed(x_pos)
        x_synset = self.synset_embed(x_synset)
        x_posi = self.position_embed([x_posi1, x_posi2])
        x_triple = self.triple_embed(x_triple)
        x_sibling = self.sibling_embed(x_sibling)

        x = self.gmlp(x)
        x_pos = self.gmlp(x_pos)
        x_synset = self.gmlp(x_synset)
        x_posi = self.gmlp(x_posi)
        x_triple = self.gmlp(x_triple)
        x_sibling = self.gmlp(x_sibling)

        x = tf.concat([x, x_sibling, x_pos, x_synset, x_posi, x_triple], axis=-1)

        x = self.to_logits(x)

        return x

