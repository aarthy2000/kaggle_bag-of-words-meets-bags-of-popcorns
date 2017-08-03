
"""
Utils for loading corpus

"""
import numpy as np

# category label to int dictionary
categories = dict([
    ('0', 0),  # negative
    ('1', 1),  # positive
    ])
category_keys = [topic for topic, index in categories.items()]
num_classes = len(category_keys)
max_vocab_count = 50000
max_doc_count = 100000


# create vocab index
def get_vocab(dict_path):
    """Create vocabulary from a file
       reserve characters: 0 (padding) 1 (OOV)
       returns: vocabulary, token_to_index
    """

    # set '[__PAD__]' & '[__OOV__]' in the sense_dict.txt

    vocab = dict()
    token_to_index = dict()

    with open(dict_path, 'r', encoding='utf8') as fin:
        lines = [line.strip() for line in fin.readlines()]

    for line in lines:
        if len(vocab) > max_vocab_count:
            break
        index, word = line.split('\t')
        if word not in token_to_index:
            token_to_index[word] = int(index)
            vocab[int(index)] = word

    print('Vocab has {} words'.format(len(vocab)))
    return vocab, token_to_index


# create vocab reverse index
def reverse_index(vocab):
    token_to_index = dict()
    for v_id, v_word in vocab.items():
        token_to_index[v_word] = v_id
    return token_to_index


def category_stats(category_labels):
    if isinstance(category_labels, np.ndarray):
        category_labels = np.argmax(category_labels, 1)
    count_per_category = dict()
    for c in category_labels:
        count_per_category[c] = count_per_category.get(c, 0) + 1
    return count_per_category


def category_stats_print(count_per_category):
    return '\n'.join(["Category %s: %s docs" % (c, count_per_category[c]) for c in count_per_category])


# load data and parse on word-level
def cnn_corpus_loader(corpus_path, dict_path, vocab=None, max_sequence_length=500,
                      random_seed=1337, max_doc_count=max_doc_count, ignore_short=True):
    """Split data and encode to vector form
       first column: category label
       remainging columns: words
    """
    if vocab is None:
        vocab, token_to_index = get_vocab(dict_path)
    else:
        token_to_index = reverse_index(vocab)

    with open(corpus_path, 'r', encoding='utf8') as fin:
        lines = [line.strip().split('\t') for line in fin]
    if len(lines) > max_doc_count:
        lines = lines[:max_doc_count]
    np.random.seed(random_seed)
    np.random.shuffle(lines)
    max_sequence_length_in_corpus = 0
    min_sequence_length_in_corpus = 999999

    cate_labels = []
    tags = []
    for line in lines:
        raw_cate_label, raw_tags = line[1], line[2].split(' ')
        # we ignore the too short data, but this constraint only exist in TRAINING
        if ignore_short and len(raw_tags) < 100:
            continue
        max_sequence_length_in_corpus = max(len(raw_tags), max_sequence_length_in_corpus)
        min_sequence_length_in_corpus = min(len(raw_tags), min_sequence_length_in_corpus)

        # tag = [token_to_index[w] for w in raw_tags if w in token_to_index]
        tag = raw_tags
        if len(tag) > max_sequence_length:
            tag = tag[:max_sequence_length]
        if len(tag) < max_sequence_length:
            # pad right side
            tag += [0] * (max_sequence_length - len(tag))
            # pad left side
            # tag = [0] * (max_sequence_length - len(tag)) + tag
        tags.append(tag)
        cate_label = [0] * num_classes
        cate_label[categories[raw_cate_label]] = 1
        cate_labels.append(cate_label)
    print("Max sequence length in corpus: %s" % max_sequence_length_in_corpus)
    print("Min sequence length in corpus: %s" % min_sequence_length_in_corpus)

    return np.array(tags, dtype=np.int16), np.array(cate_labels, dtype=np.int16), vocab, token_to_index


def get_corpus_hist(filename):
    """
    get corpus histogram
    """
    with open(filename, 'r') as fin:
        lines = [l.strip().split()[1:] for l in fin.readlines()]
    lengths = [len(l) for l in lines]
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(lengths, 20, facecolor='green', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Freq')
    plt.xticks(np.arange(0, 6000, 250), rotation=45)
    plt.title('Histogram')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('hist_' + filename + '.pdf')
    plt.close()


if __name__ == '__main__':
    # set train & test file path
    dict_path = 'data/word2vec/word_dict.txt'
    train_path = 'dataOutput/validation/labeledTrainData_clean_training_encoded.tsv'
    max_sequence_length = 3000
    random_seed = 1337

    train_x, train_y, train_vocabulary, train_token_to_index = cnn_corpus_loader(corpus_path=train_path,
                                                                                 dict_path=dict_path,
                                                                                 vocab=None,
                                                                                 max_sequence_length=max_sequence_length,
                                                                                 random_seed=random_seed)
