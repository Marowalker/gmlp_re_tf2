import codecs
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
import tensorflow as tf


class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.

        FIX: Have you tried running python build_data first?
        This will build vocab file from your train, test and dev sets and
        trim your word vectors.""".format(filename)

        super(MyIOError, self).__init__(message)


def get_trimmed_w2v_vectors(filename):
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def load_vocab(filename):
    try:
        d = dict()
        with codecs.open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx + 1  # preserve idx 0 for pad_tok

    except IOError:
        raise MyIOError(filename)
    return d


def max_count_ent(entities):
    temp = []

    for ent in entities:
        if str(-1) in ent:
            entities.remove(ent)
        elif 't' in ent:
            temp.append(ent)
        else:
            if entities.count(ent) == max([entities.count(d) for d in entities]):
                if ent not in temp:
                    temp.append(ent)
    return temp


def load_most_freq_entities():
    file = open('data/cdr_data/cdr_test.txt')
    lines = file.readlines()
    entity_dict = defaultdict(list)
    most_frequent_ent = defaultdict()
    for line in lines:
        tokens = line.split('\t')
        if len(tokens) == 1:
            title = tokens[0].split('|')
            if 't' in title:
                title_len = len(title[-1])
            else:
                pass
        else:
            # print(title_len)
            if tokens[-2] == 'Chemical':
                if int(tokens[2]) <= title_len - 1:
                    entity_dict[tokens[0]].append(tuple([tokens[-1].strip(), 't']))
                else:
                    entity_dict[tokens[0]].append(tuple([tokens[-1].strip(), 'a']))
            else:
                pass

    for abstract in entity_dict:
        max_ent = max_count_ent(entity_dict[abstract])
        most_frequent_ent[abstract] = max_ent
    return most_frequent_ent


def countNumRelation():
    with open('./data/all_depend.txt', 'r') as f:
    # with open('./data/no_dir_depend.txt', 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count


def countNumPos():
    with open('./data/all_pos.txt', 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count


def countNumSynset():
    with open('./data/all_hypernyms.txt', 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count


def countNumChar():
    with open('./data/characters.txt', 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count


def countNumTree():
    with open('./data/all_syntactic_tree.txt', 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count


def countVocab():
    with open('./data/vocab_lower.txt', 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count


def make_triple_vocab(infile):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    # raw_vocab = defaultdict()
    id_vocab = defaultdict()
    for idx, line in enumerate(lines):
        if idx != 0:
            pairs = line.split('\t')
            name, name_id = pairs[0], pairs[1]
            # raw_vocab[name_id] = name
            id_vocab[name_id] = idx
    return id_vocab


def gen(values, labels, seq_len):
    def iter():
        words, poses = values
        for i in range(len(words)):
            ws = np.array(words[i], dtype="int32")
            ps = np.array(poses[i], dtype='int32')
            label = np.array(labels[i], dtype="int32")

            review_length = ws.shape[0]
            if review_length < seq_len:
                pad_length = seq_len - review_length
                ws = np.pad(ws, (0, pad_length), constant_values=0.)
                ps = np.pad(ps, (0, pad_length), constant_values=0.)
            elif review_length > seq_len:
                ws = ws[:seq_len]
                ps = ps[:seq_len]

            yield ws, ps, label

    return iter


def make_dataset(values, labels, seq_len, batch_size):
    ds_args = ((tf.int64, tf.int64, tf.int64), (tf.TensorShape([seq_len]), tf.TensorShape([seq_len]), tf.TensorShape([2])))
    ds = tf.data.Dataset.from_generator(gen(values, labels, seq_len), *ds_args)
    # ds = ds.shuffle(buffer_size=100000, seed=1234)
    ds = ds.batch(batch_size)
    return ds
