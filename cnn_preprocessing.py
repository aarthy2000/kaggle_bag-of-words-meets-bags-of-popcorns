import random
import re
from corpus_loader import load_training_data


def generate_validation_set(data_path, split_ratio, output_dir, output_file_prefix):
    lines = load_training_data(file_path=data_path)

    num_reviews = len(lines)
    random.shuffle(lines)

    training_list = lines[:int(num_reviews * split_ratio)]
    validation_list = lines[int(num_reviews * split_ratio):]

    with open(output_dir + output_file_prefix + '_training.tsv', 'w', encoding='utf8', newline='\n') as training_out:
        for id, sentiment, review in training_list:
            training_out.write(id + '\t' + sentiment + '\t' + review + '\n')

    with open(output_dir + output_file_prefix + '_validation.tsv', 'w', encoding='utf8', newline='\n') as validation_out:
        for id, sentiment, review in validation_list:
            validation_out.write(id + '\t' + sentiment + '\t' + review + '\n')


def indexing_w2v(w2v_file_path, word_dict_path):
    word_dict = list()
    word_dict.append('[__PAD__]')  # padding
    word_dict.append('[__OOV__]')  # out of vocabulary

    # load word2vec data
    with open(w2v_file_path, 'r', encoding='utf8') as w2v_in:
        lines = [line for line in w2v_in]

    # skip first line
    for w2v_line in lines[1:]:
        word, _ = w2v_line.split(' ', 1)
        word_dict.append(word)

    with open(word_dict_path, 'w', encoding='utf8', newline='\n') as dict_out:
        for index, word in enumerate(word_dict):
            dict_out.write(str(index) + '\t' + word + '\n')


def encode_data(data_path, word_dict_path, encoded_data_path):
    data_lines = load_training_data(data_path)

    # load word dict and make dict
    word_dict = dict()
    with open(word_dict_path, 'r', encoding='utf8') as word_dict_in:
        lines = [line.strip().split('\t') for line in word_dict_in]
    for index, word in lines:
        word_dict[word] = index

    output_lines = list()
    for id, sentiment, review in data_lines:
        line = id + '\t' + sentiment + '\t'
        assert len(review) > 0, 'len(review) is {}'.format(len(review))
        review_sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', review)  # split to sentences
        for sentence in review_sentences:
            for word in sentence.split(' '):
                if word not in word_dict:
                    word = '[__OOV__]'
                line += word_dict[word] + ' '
            # padding between two sentences
            line += word_dict['[__PAD__]'] + ' '
        line = line.strip()
        line += '\n'
        output_lines.append(line)

    with open(encoded_data_path, 'w', encoding='utf8', newline='\n') as encode_out:
        encode_out.writelines(output_lines)


if __name__ == '__main__':
    # # generate validation set
    # data_path = 'dataOutput/labeledTrainData_clean.tsv'
    # split_ratio = 0.9
    # output_dir = 'dataOutput/validation/'
    # output_file_prefix = 'labeledTrainData_clean'
    # generate_validation_set(data_path, split_ratio, output_dir, output_file_prefix)

    # # indexing all words in word2vec
    # w2v_file_path = 'data/word2vec/labeledTrainData_sentences_word2vec.txt'
    # word_dict_path = 'data/word2vec/word_dict.txt'
    # indexing_w2v(w2v_file_path, word_dict_path)

    # encode all the words of reviews in training data and validation data
    data_path = 'dataOutput/validation/labeledTrainData_clean_validation.tsv'
    word_dict_path = 'data/word2vec/word_dict.txt'
    encoded_data_path = 'dataOutput/validation/labeledTrainData_clean_validation_encoded.tsv'
    encode_data(data_path, word_dict_path, encoded_data_path)




