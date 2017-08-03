from corpus_loader import load_training_data
from corpus_loader import load_testing_data


def count_label(file_path):
    lines = load_training_data(file_path=file_path)

    pos_count = 0
    neg_count = 0

    for id, sentiment, review in lines:
        # skip first line
        if id == 'id' and sentiment == 'sentiment' and review == 'review':
            continue

        if sentiment == '1':
            pos_count += 1
        elif sentiment == '0':
            neg_count += 1

    print(pos_count)  # 12500
    print(neg_count)  # 12500


def max_min_length(review_lines):
    max_length = -1
    min_length = 999999

    for review in review_lines:
        review_word_length = len(review.split(' '))
        max_length = max(max_length, review_word_length)
        min_length = min(min_length, review_word_length)

    print('max length of review is {}'.format(max_length))
    print('min length of review is {}'.format(min_length))
    # dataOutput/labeledTrainData_clean.tsv
    # max length of review is 2738 (words)
    # min length of review is 11 (words)

    # dataOutput/testData_clean.tsv
    # WARNING: test
    # max length of review is 2595 (words)
    # min length of review is 8 (words)


if __name__ == '__main__':
    # count_label('data/labeledTrainData.tsv')

    file_path = 'dataOutput/labeledTrainData_clean.tsv'
    lines = load_training_data(file_path=file_path)
    # lines = load_testing_data(file_path=file_path)
    max_min_length(review_lines=[line[-1] for line in lines])
