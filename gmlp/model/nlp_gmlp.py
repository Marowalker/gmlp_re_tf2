import tensorflow as tf
from tensorflow.keras import Model, Sequential

import constants
from gmlp.gmlp import gMLP
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense, Flatten
from gmlp.custom_layers import *
from data_utils import *


class NLPgMLPModel(Model):
    def __init__(self,
                 depth,
                 embedding_dim,
                 seq_len,
                 embeddings,
                 triples,
                 wordnet,
                 **kwargs):
        super(NLPgMLPModel, self).__init__()

        # self.to_embed = Embedding(num_tokens + 1, embedding_dim, input_length=seq_len, weights=[embeddings])
        #
        # self.pos_embed = Embedding(num_pos + 1, pos_dim, input_length=seq_len)
        self.num_of_words = countVocab()
        self.num_of_siblings = self.num_of_words
        self.num_of_poses = countNumPos()
        self.num_of_depend = countNumRelation()
        self.num_of_synsets = countNumSynset()

        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

        # self.to_embed = CustomEmbedding(name='word_embedding', dropout=constants.DROPOUT, num_depend=self.num_of_depend,
        #                                 pretrained=embeddings)
        # self.pos_embed = CustomEmbedding(name='pos_embedding', dropout=constants.DROPOUT, num_depend=self.num_of_depend,
        #                                  dim=[self.num_of_poses + 1, 6])
        # self.synset_embed = CustomEmbedding('synset', constants.DROPOUT, self.num_of_depend,
        #                                     pretrained=wordnet)
        embedding_rel = tf.Variable(self.initializer(shape=[self.num_of_depend, embedding_dim],
                                                     dtype=tf.float32), name="re_lut", trainable=True)
        word_embedding = tf.concat([embeddings, embedding_rel], axis=0)
        self.to_embed = Embedding(self.num_of_words + self.num_of_depend + 1, embedding_dim, input_length=seq_len,
                                  weights=[word_embedding])

        self.pos_embed = Embedding(self.num_of_poses + self.num_of_depend + 1, 6, input_length=seq_len)

        synset_rel_emb = tf.Variable(self.initializer(shape=[self.num_of_depend, 18]),
                                     dtype=tf.float32, trainable=True)
        synset_emb = tf.concat([wordnet, synset_rel_emb], axis=0)
        self.synset_embed = Embedding(self.num_of_synsets + self.num_of_depend + 1, 18, input_length=seq_len,
                                      weights=[synset_emb])

        self.sibling_embed = SiblingEmbedding(constants.DROPOUT, self.num_of_words, self.num_of_depend)

        # p_sibling = tf.nn.pool(all_sb_rel_lookup, window_shape=[16, 1], pooling_type="MAX", padding="SAME")

        self.position_embed = PositionEmbedding(constants.DROPOUT, self.num_of_depend, [seq_len * 2, 25])

        # self.triple_embed = TripleEmbedding(constants.DROPOUT, triples)
        self.triple_embed = Embedding(triples.shape[0], embedding_dim, input_length=seq_len, weights=[triples])

        self.gmlp = gMLP(
            dim=embedding_dim,
            depth=depth,
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.gmlp_sibling = gMLP(
            dim=embedding_dim,
            depth=depth,
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.gmlp_pos = gMLP(
            dim=6,
            depth=depth,
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.gmlp_synsets = gMLP(
            dim=wordnet.shape[-1],
            depth=depth,
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.gmlp_position = gMLP(
            dim=50,
            depth=depth,
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.gmlp_triple = gMLP(
            dim=embedding_dim,
            depth=depth,
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.to_logits = Sequential([
            Flatten(data_format="channels_first"),
            LayerNormalization(),
            Dense(128),
            Dense(128),
            Dense(2, activation="softmax")
        ])

    def call(self, inputs, training=None, mask=None):
        x, x_pos, x_synset, x_posi1, x_posi2, x_triple, x_sibling = inputs
        # x, x_pos, x_synset, x_triple = inputs
        # x = tf.cast(inputs[0], dtype="int64")
        # x_pos = tf.cast(inputs[-1], dtype='int64')

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

        # x = tf.concat([x, x_sibling, x_pos, x_synset, x_posi, x_triple], axis=-1)
        x = self.gmlp(x)
        x_pos = self.gmlp_pos(x_pos)
        x_synset = self.gmlp_synsets(x_synset)
        x_posi = self.gmlp_position(x_posi)
        x_triple = self.gmlp_triple(x_triple)
        x_sibling = self.gmlp_sibling(x_sibling)
        x = tf.concat([x, x_sibling, x_pos, x_synset, x_posi, x_triple], axis=-1)
        # x = tf.concat([x, x_pos, x_synset, x_triple], axis=-1)

        x = self.to_logits(x)
        return x
