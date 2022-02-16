from nltk.corpus import wordnet as wn
import constants
from data_utils import *
from dataset import Dataset, parse_raw, parse_words, pad_sequences

# with open('data/raw_data/sdp_data_acentors_triples.train.txt', 'r') as f:
#     lines = f.readlines()
#
# train_words, train_poses, train_labels, train_ids = parse_words(lines)
# print(train_poses)
# data_words, data_siblings, data_postitions, data_y, data_pos, data_synsets, data_relations, data_directions, \
#     identities = parse_raw(lines)

# print(data_words)


vocab_words = load_vocab(constants.ALL_WORDS)
vocab_poses = load_vocab(constants.ALL_POSES)
vocab_synsets = load_vocab(constants.ALL_SYNSETS)
vocab_depends = load_vocab(constants.ALL_DEPENDS)

vocab_chems = make_triple_vocab(constants.DATA + 'chemical2id.txt')
vocab_dis = make_triple_vocab(constants.DATA + 'disease2id.txt')
vocab_rel = make_triple_vocab(constants.DATA + 'relation2id.txt')
#
train = Dataset('data/raw_data/sdp_data_acentors_triples.train.txt', 'data/raw_data/sdp_triple.train.txt',
                vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets, vocab_chems=vocab_chems,
                vocab_dis=vocab_dis, vocab_rel=vocab_rel, vocab_depends=vocab_depends)

test = Dataset('data/raw_data/sdp_data_acentors_triples.test.txt', 'data/raw_data/sdp_triple.test.txt',
               vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets, vocab_chems=vocab_chems,
               vocab_dis=vocab_dis, vocab_rel=vocab_rel, vocab_depends=vocab_depends)

# print(np.array(train.siblings))
# print('\n')
# print(np.array(train.words))
#
# train_dummy = DummyDataset('data/raw_data/sdp_data_acentors_triples.train.txt', vocab_words=vocab_words,
#                            vocab_poses=vocab_poses)

train_tf = train.to_tf(constants.MAX_LENGTH)
for elem in train_tf['words']:
    print(elem)

# print('\n')

# siblings, _ = pad_sequences(train.siblings, pad_tok=0, max_sent_length=constants.MAX_LENGTH, nlevels=2)
# print(siblings.shape)

# train_data = make_dataset(values=[train_dummy.words, train_dummy.poses], labels=train_dummy.labels,
#                           seq_len=constants.MAX_LENGTH,
#                           batch_size=256)
# train_data = train_data.batch(256)

# for elem in train_data:
#     print(elem)
# print(train_dummy.poses)

# print(len(train.words))
# print(len(train.labels))

# train_data = train_dummy.to_tf(seq_len=constants.MAX_LENGTH, batch_size=256)
# for step, (word_batch_train, pos_batch_train, y_batch_train) in enumerate(train_data):
#     print(y_batch_train)

# with open('data/raw_data/sdp_data_acentors_triples.train.txt', 'r') as f:
#     raw_data = f.readlines()
#
# raw_words, raw_labels, raw_identities = parse_words(raw_data)
# print(len(raw_labels))
