import pickle
from data_utils import *
from dataset import Dataset
from evaluate.bc5 import evaluate_bc5
from gmlp.model.nlp_gmlp import NLPgMLPModel
from tensorflow.keras.optimizers import Adam
from gmlp.custom_layers import *


LEARNING_RATE = 1e-4
BATCH_SIZE = 128

np.random.seed(13)


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


def main():
    result_file = open('data/results.txt', 'a')

    # Get word embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
    wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')

    vocab_words = load_vocab(constants.ALL_WORDS)
    vocab_poses = load_vocab(constants.ALL_POSES)
    vocab_synsets = load_vocab(constants.ALL_SYNSETS)
    vocab_depends = load_vocab(constants.ALL_DEPENDS)

    vocab_chems = make_triple_vocab(constants.DATA + 'chemical2id.txt')
    vocab_dis = make_triple_vocab(constants.DATA + 'disease2id.txt')
    vocab_rel = make_triple_vocab(constants.DATA + 'relation2id.txt')

    with open('data/w2v_model/triple_embeddings.pkl', 'rb') as f:
        triple_emb = pickle.load(f)

    for i in range(1):
        if constants.IS_REBUILD == 1:
            print('Build data')
            train = Dataset('data/raw_data/sdp_data_acentors_triples.train.txt', 'data/raw_data/sdp_triple.train.txt',
                            vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                            vocab_chems=vocab_chems,
                            vocab_dis=vocab_dis, vocab_rel=vocab_rel, vocab_depends=vocab_depends)
            pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
            dev = Dataset('data/raw_data/sdp_data_acentors_triples.dev.txt', 'data/raw_data/sdp_triple.dev.txt',
                          vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                          vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                          vocab_depends=vocab_depends)
            pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
            test = Dataset('data/raw_data/sdp_data_acentors_triples.test.txt', 'data/raw_data/sdp_triple.test.txt',
                           vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                           vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                           vocab_depends=vocab_depends)
            pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            print('Load data')
            train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
            dev = pickle.load(open(constants.PICKLE_DATA + 'dev.pickle', 'rb'))
            test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))

        # Train, Validation Split
        validation = Dataset('', '', process_data=False)
        train_ratio = 0.85
        n_sample = int(len(dev.words) * (2 * train_ratio - 1))
        props = ['words', 'siblings', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations',
                 'directions', 'identities', 'triples']
        # props = ['words', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'triples', 'identities']
        # props = ['words', 'poses', 'labels', 'identities']
        for prop in props:
            train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
            validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

        print("Train shape: ", len(train.words))
        print("Test shape: ", len(test.words))
        print("Validation shape: ", len(validation.words))

        train_dict = train.to_tf(constants.MAX_LENGTH)
        val_dict = validation.to_tf(constants.MAX_LENGTH, do_shuffle=False)
        test_dict = test.to_tf(constants.MAX_LENGTH, do_shuffle=False)

        with tf.device('/device:GPU:0'):
            # callback = tf.keras.callbacks.EarlyStopping(monitor='val_custom_f1', patience=10)
            callback = CustomCallback(patience=10)

            model = NLPgMLPModel(
                depth=5,
                embedding_dim=constants.INPUT_W2V_DIM,
                seq_len=constants.MAX_LENGTH,
                wordnet=wn_emb,
                causal=True,
                ff_mult=4,
                embeddings=embeddings,
                triples=triple_emb
            )

            model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy', custom_f1], run_eagerly=True)

            model.fit(
                x=(train_dict['words'], train_dict['poses'], train_dict['synsets'], train_dict['position_1'],
                   train_dict['position_2'], train_dict['triples'], train_dict['siblings']),
                y=train_dict['labels'],
                batch_size=256,
                validation_data=((val_dict['words'], val_dict['poses'], val_dict['synsets'], val_dict['position_1'],
                                  val_dict['position_2'], val_dict['triples'], val_dict['siblings']), val_dict['labels']),
                epochs=constants.EPOCHS,
                callbacks=[callback]
            )
            # model.fit(
            #     x=(train_dict['words'], train_dict['poses'], train_dict['synsets'], train_dict['triples']),
            #     y=train_dict['labels'],
            #     batch_size=256,
            #     validation_data=((val_dict['words'], val_dict['poses'], val_dict['synsets'], val_dict['triples']),
            #                      val_dict['labels']),
            #     epochs=constants.EPOCHS,
            #     callbacks=[callback]
            # )

        y_pred = []
        identities = test.identities
        answer = {}

        logits = model.predict(x=(test_dict['words'], test_dict['poses'], test_dict['synsets'],
                                  test_dict['position_1'], test_dict['position_2'], test_dict['triples'],
                                  test_dict['siblings']))
        for logit in logits:
            y_pred.append(int(np.argmax(logit)))

        # y_pred = model.predict()
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                if identities[i][0] not in answer:
                    answer[identities[i][0]] = []

                if identities[i][1] not in answer[identities[i][0]]:
                    answer[identities[i][0]].append(identities[i][1])

        # for k in answer:
        #     print(k, answer[k])

        print(
            'result: abstract: ', evaluate_bc5(answer)
        )
        result_file.write(str(evaluate_bc5(answer)))
        result_file.write('\n')


if __name__ == '__main__':
    # fix_gpu()
    main()

