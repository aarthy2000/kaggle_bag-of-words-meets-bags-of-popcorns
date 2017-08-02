import random
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


if __name__ == '__main__':
    # # generate validation set
    # data_path = 'dataOutput/labeledTrainData_clean.tsv'
    # split_ratio = 0.9
    # output_dir = 'dataOutput/validation/'
    # output_file_prefix = 'labeledTrainData_clean'
    # generate_validation_set(data_path, split_ratio, output_dir, output_file_prefix)

    # indexing all words in word2vec
    w2v_file_path = 'data/word2vec/labeledTrainData_sentences_word2vec.txt'
    word_dict_path = 'data/word2vec/word_dict.txt'
    indexing_w2v(w2v_file_path, word_dict_path)



