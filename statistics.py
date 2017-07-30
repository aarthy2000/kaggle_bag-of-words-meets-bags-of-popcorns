from corpus_loader import load_training_data


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


def max_min_length(file_path):
    lines = load_training_data(file_path=file_path)

    max_length = -1
    min_length = 999999

    for id, sentiment, review in lines:
        max_length = max(max_length, len(review))
        min_length = min(min_length, len(review))

    print('max length of review is {}'.format(max_length))  # max length of review is 13710
    print('min length of review is {}'.format(min_length))  # min length of review is 6


if __name__ == '__main__':
    # count_label('data/labeledTrainData.tsv')
    max_min_length('data/labeledTrainData.tsv')
