import numpy as np
import constants as constant
from nltk.corpus import wordnet as wn
import constants
from sklearn.utils import shuffle
import tensorflow as tf

np.random.seed(13)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, max_sent_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        # max_length = max(map(lambda x: len(x), sequences))
        max_length = max_sent_length
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        # max_length_word = max_sent_length
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        # max_length_sentence = max(map(lambda x: len(x), sequences))
        max_length_sentence = max_sent_length

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_sent_length)

    return np.array(sequence_padded), sequence_length


def np_pad(sequence, max_length=constants.MAX_LENGTH):
    pad_length = max_length - len(sequence)
    return np.pad(sequence, (0, pad_length), constant_values=0.)


def parse_raw(raw_data):
    all_words = []
    all_siblings = []
    all_positions = []
    all_relations = []
    all_directions = []
    all_poses = []
    all_labels = []
    all_synsets = []
    all_identities = []
    pmid = ''
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    siblings = []
                    positions = []
                    poses = []
                    synsets = []

                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        if idx % 2 == 0:
                            sibling_word = []
                            for idx, _node in enumerate(node):
                                word = constant.UNK if _node == '' else _node
                                if idx == 0:
                                    w, p, s = word.split('\\')
                                    p = 'NN' if p == '' else p
                                    s = str(wn.synsets('entity')[0].offset()) if s == '' else s
                                    _w, position = w.rsplit('_', 1)
                                    words.append(_w)
                                    positions.append(min(int(position), constant.MAX_LENGTH))
                                    poses.append(p)
                                    synsets.append(s)
                                else:
                                    sibling_word.append(word)
                            siblings.append(sibling_word)
                        else:
                            dependency = node[0]
                            words.append(dependency)
                            siblings.append(dependency)
                            poses.append(dependency)
                            synsets.append(dependency)
                            positions.append(dependency)

                    all_words.append(words)
                    all_siblings.append(siblings)
                    all_positions.append(positions)
                    all_poses.append(poses)
                    all_synsets.append(synsets)
                    all_labels.append([label])
                    all_identities.append((pmid, pair))
            else:
                print(l)

    return all_words, all_siblings, all_positions, all_labels, all_poses, all_synsets, all_relations, \
        all_directions, all_identities


def parse_words(raw_data):
    all_words = []
    all_poses = []
    all_labels = []
    all_identities = []
    pmid = ''
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    poses = []
                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        if idx % 2 == 0:
                            for idx, _node in enumerate(node):
                                word = constant.UNK if _node == '' else _node
                                if idx == 0:
                                    w, p, s = word.split('\\')
                                    p = 'NN' if p == '' else p
                                    s = str(wn.synsets('entity')[0].offset()) if s == '' else s
                                    _w, position = w.rsplit('_', 1)
                                    words.append(_w)
                                    poses.append(p)
                                else:
                                    w = word.split('\\')[0]
                        else:
                            pass

                    all_words.append(words)
                    all_poses.append(poses)
                    all_labels.append([label])
                    all_identities.append((pmid, pair))
            else:
                print(l)

    return all_words, all_poses, all_labels, all_identities


class Dataset:
    def __init__(self, data_name, triple_name, vocab_words=None, vocab_poses=None, vocab_synset=None,
                 vocab_depends=None, vocab_chems=None, vocab_dis=None, vocab_rel=None, process_data=True):
        self.data_name = data_name
        self.triple_name = triple_name

        self.words = None
        self.siblings = None
        self.positions_1 = None
        self.positions_2 = None
        self.labels = None
        self.poses = None
        self.synsets = None
        self.relations = None
        self.directions = None
        self.identities = None

        self.vocab_words = vocab_words
        self.vocab_poses = vocab_poses
        self.vocab_synsets = vocab_synset
        self.vocab_depends = vocab_depends

        self.vocab_chems = vocab_chems
        self.vocab_dis = vocab_dis
        self.vocab_rel = vocab_rel

        if process_data:
            self._process_data()
            self._clean_data()

    def _clean_data(self):
        del self.vocab_words
        del self.vocab_poses
        del self.vocab_synsets
        del self.vocab_depends

    def _process_data(self):
        with open(self.data_name, 'r') as f:
            raw_data = f.readlines()
        data_words, data_siblings, data_postitions, data_y, data_pos, data_synsets, data_relations, \
            data_directions, self.identities = parse_raw(raw_data)

        with open(self.triple_name, 'r') as f2:
            raw_triple = f2.readlines()

        triple_data = self.parse_triple(raw_triple)

        words = []
        siblings = []
        positions_1 = []
        positions_2 = []
        labels = []
        poses = []
        synsets = []
        relations = []
        directions = []

        for i in range(len(data_postitions)):
            position_1, position_2 = [], []
            e1 = data_postitions[i][0]
            e2 = data_postitions[i][-1]
            for po in data_postitions[i]:
                if data_postitions[i].index(po) % 2 == 0:
                    position_1.append((po - e1 + constant.MAX_LENGTH) // 5 + 1)
                    position_2.append((po - e2 + constant.MAX_LENGTH) // 5 + 1)
                else:
                    rid_ps = int(self.vocab_depends[po]) + constant.MAX_LENGTH // 5 + 1
                    position_1.append(rid_ps)
                    position_2.append(rid_ps)

            positions_1.append(position_1)
            positions_2.append(position_2)

        for i in range(len(data_words)):

            ws, sbs, ps, ss = [], [], [], []

            for w, sb, p, s in zip(data_words[i], data_siblings[i], data_pos[i], data_synsets[i]):
                if data_words[i].index(w) % 2 == 0:
                    if w in self.vocab_words:
                        word_id = self.vocab_words[w]
                    else:
                        word_id = self.vocab_words[constant.UNK]
                    ws.append(word_id)

                    temp = []
                    for token_word in sb:
                        if token_word in self.vocab_words:
                            sibling_id = self.vocab_words[token_word]
                        else:
                            sibling_id = self.vocab_words[constant.UNK]
                        # sbs.append(sibling_id)
                        temp.append(sibling_id)
                    sbs.append(temp)

                    if p in self.vocab_poses:
                        p_id = self.vocab_poses[p]
                    else:
                        p_id = self.vocab_poses['NN']
                    ps += [p_id]
                    if s in self.vocab_synsets:
                        synset_id = self.vocab_synsets[s]
                    else:
                        synset_id = self.vocab_synsets[constant.UNK]
                    ss += [synset_id]

                else:
                    rid_w = int(self.vocab_depends[w]) + len(self.vocab_words)
                    rid_p = int(self.vocab_depends[w]) + len(self.vocab_poses)
                    rid_s = int(self.vocab_depends[w]) + len(self.vocab_synsets)

                    ws.append(rid_w)
                    sbs.append([rid_w])
                    ps.append(rid_p)
                    ss.append(rid_s)

            words.append(ws)
            siblings.append(sbs)
            poses.append(ps)
            synsets.append(ss)

            # lb = constant.ALL_LABELS.index(data_y[i][0])
            if data_y[i][0] == 'CID':
                lb = [1, 0]
            else:
                lb = [0, 1]
            labels.append(lb)

        self.words = words
        self.siblings = siblings
        self.positions_1 = positions_1
        self.positions_2 = positions_2
        self.labels = labels
        self.poses = poses
        self.synsets = synsets
        self.relations = relations
        self.directions = directions
        self.triples = triple_data

    def to_tf(self, seq_len, do_shuffle=True):
        padded_words, _ = pad_sequences(self.words, pad_tok=0, max_sent_length=seq_len)
        padded_poses, _ = pad_sequences(self.poses, pad_tok=0, max_sent_length=seq_len)
        padded_synsets, _ = pad_sequences(self.synsets, pad_tok=0, max_sent_length=seq_len)
        padded_posi_1, _ = pad_sequences(self.positions_1, pad_tok=0, max_sent_length=seq_len)
        padded_posi_2, _ = pad_sequences(self.positions_2, pad_tok=0, max_sent_length=seq_len)
        padded_tripled, _ = pad_sequences(self.triples, pad_tok=0, max_sent_length=seq_len)
        padded_siblings, _ = pad_sequences(self.siblings, pad_tok=0, max_sent_length=seq_len, nlevels=2)

        if do_shuffle:
            words_shuffled, siblings_shuffled, poses_shuffled, synsets_shuffled, posi_1_shuffled, \
                posi_2_shuffled, triples_shuffled, labels_shuffled = shuffle(
                    padded_words, padded_siblings, padded_poses, padded_synsets, padded_posi_1, padded_posi_2,
                    padded_tripled,
                    self.labels, random_state=1234
            )

            tf_data = {
                'words': np.array(words_shuffled),
                'siblings': np.array(siblings_shuffled),
                'poses': np.array(poses_shuffled),
                'synsets': np.array(synsets_shuffled),
                'position_1': np.array(posi_1_shuffled),
                'position_2': np.array(posi_2_shuffled),
                'triples': np.array(triples_shuffled),
                'labels': np.array(labels_shuffled)
            }
        else:
            tf_data = {
                'words': np.array(padded_words),
                'siblings': np.array(padded_siblings),
                'poses': np.array(padded_poses),
                'synsets': np.array(padded_synsets),
                'position_1': np.array(padded_posi_1),
                'position_2': np.array(padded_posi_2),
                'triples': np.array(padded_tripled),
                'labels': np.array(self.labels)
            }

        return tf_data

    def parse_triple(self, raw_data):
        all_triples = []
        for line in raw_data:
            l = line.split()
            if len(l) == 1:
                pass
            else:
                c, d, r = l
                c_id = int(self.vocab_chems[c])
                d_id = int(self.vocab_dis[d]) + c_id
                r_id = int(self.vocab_rel[r]) + d_id
                all_triples.append([c_id, d_id, r_id, 0, 0])

        return all_triples

