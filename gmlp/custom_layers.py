import tensorflow as tf
import numpy as np
import constants
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score


class CustomEmbedding(tf.keras.Model):
    def __init__(self, name, dropout, num_depend=None, pretrained=None, dim=None):
        super(CustomEmbedding, self).__init__()
        self.emb_name = name
        self.dropout = dropout
        self.num_depend = num_depend
        self.pretrained = pretrained
        self.dim = dim
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)
        if self.pretrained is not None:
            dummy = tf.Variable(np.zeros((1, self.pretrained.shape[-1])), name='dummy', dtype=tf.float32,
                                trainable=False)
            # Create dependency relations randomly
            self.embeddings_re = tf.Variable(self.initializer(shape=[self.num_depend + 1, self.pretrained.shape[-1]],
                                                              dtype=tf.float32), name="re_lut")
            self.w = tf.Variable(self.pretrained, name=self.emb_name + '_lut', dtype=tf.float32)
            self.w = tf.concat([dummy, self.w], axis=0)
            self.w = tf.concat([self.w, self.embeddings_re], axis=0)
        else:
            dummy = tf.Variable(np.zeros((1, self.dim[-1])), name='dummy', dtype=tf.float32, trainable=False)
            # Create dependency relations randomly
            self.embeddings_re = tf.Variable(self.initializer(shape=[self.num_depend + 1, self.dim[-1]],
                                                              dtype=tf.float32), name="re_lut", trainable=True)
            # Concat dummy vector and relations vectors
            # self.embeddings_re = tf.concat([dummy, self.embeddings_re], axis=0)
            self.w = self.add_weight(shape=self.dim, name=self.emb_name + '_lut',
                                     initializer=self.initializer, regularizer=self.regularizer, trainable=True)
            self.w = tf.concat([dummy, self.w], axis=0)
            self.w = tf.concat([self.w, self.embeddings_re], axis=0)

    def call(self, inputs, **kwargs):
        lookup = tf.nn.embedding_lookup(params=self.w, ids=inputs)
        return tf.nn.dropout(lookup, self.dropout)


class TripleEmbedding(tf.keras.layers.Layer):
    def __init__(self, dropout, triple):
        super(TripleEmbedding, self).__init__()
        self.dropout = dropout
        self.triple = triple

    def build(self, input_shape):
        self.w = tf.Variable(self.triple, name='triple_lut', dtype=tf.float32, trainable=False)

    def call(self, inputs, **kwargs):
        lookup = tf.nn.embedding_lookup(params=self.w, ids=inputs)
        return tf.nn.dropout(lookup, self.dropout)


class SiblingEmbedding(tf.keras.layers.Layer):
    def __init__(self, dropout, num_of_siblings, num_of_depend):
        super(SiblingEmbedding, self).__init__()
        self.dropout = dropout
        self.num_of_siblings = num_of_siblings
        self.num_of_depend = num_of_depend
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

    def build(self, input_shape):
        dummy = tf.Variable(np.zeros((1, constants.INPUT_W2V_DIM)), name='dummy', dtype=tf.float32,
                            trainable=False)
        # Create dependency relations randomly
        self.embeddings_re = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, constants.INPUT_W2V_DIM],
                                                          dtype=tf.float32), name="re_lut")
        self.embeddings_re = tf.concat([dummy, self.embeddings_re], axis=0)
        self.embeddings_sb = self.add_weight(shape=[self.num_of_siblings + 1, 15], dtype=tf.float32, name="sb_lut",
                                             initializer=self.initializer, regularizer=self.regularizer, trainable=True)
        dummy_eb_ex = tf.Variable(np.zeros((self.num_of_siblings + 1, constants.INPUT_W2V_DIM - 15)),
                                  name="dummy_ex", dtype=tf.float32,
                                  trainable=False)
        self.embeddings_sb = tf.concat([self.embeddings_sb, dummy_eb_ex], axis=-1)

        # p_sibling = tf.nn.pool(all_sb_rel_lookup, window_shape=[16, 1], pooling_type="MAX", padding="SAME")

        self.w = tf.Variable(self.initializer(shape=[1, constants.INPUT_W2V_DIM], dtype=tf.float32),
                             name="sibling_weight", trainable=True)

    def call(self, inputs, **kwargs):
        all_sb_rel_table = tf.concat([self.embeddings_re, self.embeddings_sb], axis=0)
        all_sb_rel_lookup = tf.nn.embedding_lookup(params=all_sb_rel_table, ids=inputs)

        all_sb_mean = all_sb_rel_lookup * self.w

        # p_sibling = tf.reduce_max(p_sibling, axis=2)
        p_sibling = tf.reduce_mean(input_tensor=all_sb_mean, axis=2)
        return tf.nn.dropout(p_sibling, 1 - self.dropout)


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, dropout, num_of_depend, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.dropout = dropout
        self.num_of_depend = num_of_depend
        self.embedding_dim = embedding_dim
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

    def build(self, input_shape):
        self.embeddings_position = self.add_weight(shape=self.embedding_dim, dtype=tf.float32,
                                                   initializer=self.initializer, regularizer=self.regularizer,
                                                   name='position_lut', trainable=True)
        dummy_posi_emb = tf.Variable(np.zeros((1, self.embedding_dim[-1])),
                                     dtype=tf.float32)  # constants.INPUT_W2V_DIM // 2)), dtype=tf.float32)
        self.embeddings_position = tf.concat([dummy_posi_emb, self.embeddings_position], axis=0)
        dummy_eb3 = tf.Variable(np.zeros((1, 50)), name="dummy3", dtype=tf.float32, trainable=False)

        embeddings_re3 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, (self.embedding_dim[-1] * 2)],
                                                      dtype=tf.float32), name="re_lut3")
        embeddings_re3 = tf.concat([dummy_eb3, embeddings_re3], axis=0)
        # Concat each position vector with half of each dependency relation vector
        self.embeddings_position1 = tf.concat([self.embeddings_position, embeddings_re3[:, :self.embedding_dim[-1]]],
                                              axis=0)  # :int(constants.INPUT_W2V_DIM / 2)]], axis=0)
        self.embeddings_position2 = tf.concat([self.embeddings_position, embeddings_re3[:, self.embedding_dim[-1]:]],
                                              axis=0)  # int(constants.INPUT_W2V_DIM / 2):]], axis=0)

    def call(self, inputs, *args, **kwargs):
        position_1 = tf.nn.embedding_lookup(params=self.embeddings_position1, ids=inputs[0])
        position_1 = tf.nn.dropout(position_1, self.dropout)
        position_2 = tf.nn.embedding_lookup(params=self.embeddings_position2, ids=inputs[-1])
        position_2 = tf.nn.dropout(position_2, self.dropout)
        position_embeddings = tf.concat([position_1, position_2], axis=-1)
        return position_embeddings


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def custom_f1(y_true, y_pred):
    accuracy = []
    f1 = []
    predict = []
    exclude_label = []
    logits = y_pred.numpy()
    labels = y_true.numpy()
    for logit, label in zip(logits, labels):
        logit = np.argmax(logit)
        label = np.argmax(label)
        exclude_label.append(label)
        predict.append(logit)
        accuracy += [logit == label]

    f1.append(f1_score(predict, exclude_label, average='macro'))
    return np.mean(f1)


class CustomCallback(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(CustomCallback, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_custom_f1")
        if np.greater(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
