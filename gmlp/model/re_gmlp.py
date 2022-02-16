import tensorflow as tf
from tensorflow.keras import Model, Sequential
from gmlp.gmlp import gMLP
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense, Flatten
import constants
from data_utils import *
from gmlp.custom_layers import *


class REgMLP(Model):
    def __init__(self, embeddings, triples, wordnet, depth):
        super(REgMLP, self).__init__()

        self.embeddings = embeddings
        self.triples = triples
        self.wordnet_emb = wordnet

        self.max_length = constants.MAX_LENGTH
        # Num of dependency relations
        self.num_of_depend = countNumRelation()
        # Num of pos tags
        self.num_of_pos = countNumPos()
        self.num_of_synset = countNumSynset()
        self.num_of_words = countVocab()
        self.num_of_class = len(constants.ALL_LABELS)
        self.trained_models = constants.TRAINED_MODELS
        self.initializer = tf.initializers.glorot_normal()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # self.embedding_wd = CustomEmbedding('word', constants.DROPOUT, self.num_of_depend, pretrained=self.embeddings)
        # self.embedding_pos = CustomEmbedding('pos', constants.DROPOUT, self.num_of_depend, dim=[self.num_of_pos + 1, 6])
        # self.embedding_synset = CustomEmbedding('synset', constants.DROPOUT, self.num_of_depend,
        #                                         pretrained=self.wordnet_emb)
        # self.embedding_positions = PositionEmbedding(constants.DROPOUT, self.num_of_depend, [self.max_length * 2, 25])
        # self.embedding_triples = TripleEmbedding(constants.DROPOUT, self.triples)
        self.embedding_wd = Embedding(self.num_of_words + 1, constants.INPUT_W2V_DIM, input_length=self.max_length,
                                      weights=[self.embeddings], trainable=False)

        self.gmlp_word = gMLP(
            dim=constants.INPUT_W2V_DIM,
            depth=depth,
            seq_len=self.max_length,
            activation=tf.nn.swish
        )
        # self.gmlp_pos = gMLP(
        #     dim=6,
        #     depth=depth,
        #     seq_len=self.max_length,
        #     activation=tf.nn.swish
        # )
        # self.gmlp_synset = gMLP(
        #     dim=self.wordnet_emb.shape[-1],
        #     depth=depth,
        #     seq_len=self.max_length,
        #     activation=tf.nn.swish
        # )
        # self.gmlp_position = gMLP(
        #     dim=50,
        #     depth=depth,
        #     seq_len=self.max_length,
        #     activation=tf.nn.swish
        # )
        # self.gmlp_triple = gMLP(
        #     dim=self.embeddings.shape[-1],
        #     depth=depth,
        #     seq_len=self.max_length,
        #     activation=tf.nn.swish
        # )

        self.to_logits = Sequential([
            Flatten(data_format="channels_first"),
            LayerNormalization(),
            Dense(1, activation="softmax")
        ])

    def call(self, inputs, training=None, mask=None):
        word_emb = self.embedding_wd(inputs['words'])
        # pos_emb = self.embedding_pos(inputs['poses'])
        # synset_emb = self.embedding_synset(inputs['synsets'])
        # position_emb = self.embedding_positions([inputs['positions_1'], inputs['positions_2']])
        # triple_emb = self.embedding_triples(inputs['triples'])

        word_gmlp = self.gmlp_word(word_emb)
        # pos_gmlp = self.gmlp_pos(pos_emb)
        # synset_gmlp = self.gmlp_synset(synset_emb)
        # position_gmlp = self.gmlp_position(position_emb)
        # triple_gmlp = self.gmlp_triple(triple_emb)
        #
        # all_gmlp = tf.concat([word_gmlp, pos_gmlp, synset_gmlp, position_gmlp, triple_gmlp], axis=-1)

        # outputs = self.to_logits(all_gmlp)
        outputs = self.to_logits(word_gmlp)
        return outputs
