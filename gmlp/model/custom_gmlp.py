import tensorflow as tf
import os
from data_utils import *
import constants
from gmlp.custom_layers import *
from gmlp.gmlp import gMLP
from utils import Timer, Log


class gMLPModel:
    def __init__(self, model_path, depth, embeddings, batch_size, **kwargs):
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.depth = depth
        self.embeddings = embeddings
        self.batch_size = batch_size

        self.max_length = constants.MAX_LENGTH
        self.num_of_words = countVocab()
        # Num of pos tag
        self.num_of_pos = countNumPos()
        self.num_of_classes = len(constants.ALL_LABELS)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.train_acc_metric = tf.keras.metrics.Accuracy()
        self.val_acc_metric = tf.keras.metrics.Accuracy()

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.word_ids = tf.keras.Input(name='word_ids', shape=(None,), dtype='int32')
        # Indexes of second channel (pos tags + dependency relations)
        self.pos_ids = tf.keras.Input(name='pos_ids', shape=(None,), dtype='int32')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        # initialize word embeddings
        embedding_wd = CustomEmbedding('word', constants.DROPOUT, pretrained=self.embeddings)
        self.word_embeddings = embedding_wd(self.word_ids)

        # initialize pos embeddings
        embedding_pos = CustomEmbedding('pos', constants.DROPOUT, dim=[self.num_of_pos + 1, 6])
        self.pos_embeddings = embedding_pos(self.pos_ids)

    def _siongle_channel_gmlp(self):
        all_embeddings = tf.concat([self.word_embeddings, self.pos_embeddings], axis=-1)
        gmlp = gMLP(dim=self.embeddings.shape[-1] + 6, depth=self.depth, seq_len=constants.MAX_LENGTH,
                    activation=tf.nn.swish)(all_embeddings)

        gmlp_output = tf.keras.layers.Flatten(data_format="channels_first")(gmlp)

        return gmlp_output

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        final_cnn_output = self._siongle_channel_gmlp()
        hidden_1 = tf.keras.layers.Dense(
            units=128, name="hidden_1",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(final_cnn_output)
        hidden_2 = tf.keras.layers.Dense(
            units=128, name="hidden_2",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_1)
        self.outputs = tf.keras.layers.Dense(
            units=self.num_of_classes,
            activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_2)

        self.model = tf.keras.Model(inputs=[self.word_ids, self.pos_ids], outputs=self.outputs)

    def build(self, train_data, val_data, test_data):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholders()
        self._load_data(train_data, val_data, test_data)
        self._add_word_embeddings_op()
        self._add_logits_op()

        timer.stop()

    def _load_data(self, train_data, val_data, test_data):
        timer = Timer()
        timer.start("Loading data into model...")

        self.dataset_train = train_data
        self.dataset_val = val_data
        self.dataset_test = test_data

        # print("Number of training examples:", len(self.dataset_train['labels']))
        # print("Number of validation examples:", len(self.dataset_val['labels']))
        # print("Number of test examples:", len(self.dataset_test['labels']))

        # self.dataset_train = self.dataset_train.batch(self.batch_size)
        # self.dataset_val = self.dataset_val.batch(self.batch_size)
        # self.dataset_test = self.dataset_test.batch(self.batch_size)

        timer.stop()

    @tf.function
    def train_step(self, x_word, x_pos, y):
        with tf.GradientTape() as tape:
            logits = self.model([x_word, x_pos], training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x_word, x_pos, y):
        val_logits = self.model([x_word, x_pos], training=False)
        self.val_acc_metric.update_state(y, val_logits)

    def train(self, epochs, early_stopping=False, patience=constants.PATIENCE):
        timer = Timer()
        timer.start("Training model...")

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (word_batch_train, pos_batch_train, y_batch_train) in enumerate(self.dataset_train):
                loss_value = self.train_step(word_batch_train, pos_batch_train, y_batch_train)
                # Log every 200 batches.
                if step % 500 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * self.batch_size))

            # Display metrics at the end of each epoch.
            train_acc = self.train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for word_batch_val, pos_batch_train, y_batch_val in self.dataset_val:
                self.test_step(word_batch_val, pos_batch_train, y_batch_val)

            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))

        timer.stop()

    def predict(self):
        y_pred = []
        for word_batch_test, pos_batch_test, y_batch_test in self.dataset_test:
            test_logits = self.model([word_batch_test, pos_batch_test], training=False)
            for logit in test_logits:
                decode_sequence = np.argmax(logit)
                y_pred.append(decode_sequence)
        return y_pred











