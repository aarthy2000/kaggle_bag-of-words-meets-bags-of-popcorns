

def load_training_data(file_path):
    # skip first line
    first_line = 'id	sentiment	review'
    with open(file_path, 'r', encoding='utf8') as data_in:
        lines = [line.strip().split('\t') for line in data_in if first_line not in line]
    return lines


def load_testing_data(file_path):
    # skip first line
    first_line = 'id	review'
    with open(file_path, 'r', encoding='utf8') as data_in:
        lines = [line.strip().split('\t') for line in data_in if first_line not in line]
    return lines


if __name__ == '__main__':
    training_lines = load_training_data('data/labeledTrainData.tsv')
    # test_lines = load_testing_data('data/testData.tsv')
    print(len(training_lines))
    print(training_lines[1][2].strip('"').replace('\\"', '"'))

