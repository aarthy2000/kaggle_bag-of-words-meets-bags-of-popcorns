import re
from corpus_loader import *


def make_w2v_data(reviews):
    sentences = list()
    for review in reviews:
        review = re.sub('(<br \/>)+', ' ', review)  # replace <br /> to space
        review = review.replace('\\"', '"')
        review_sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', review)  # split to sentences
        # append \n and remove space
        review_sentences = [(sentence + '\n').strip(' ') for sentence in review_sentences]
        print(len(review_sentences))
        sentences.extend(review_sentences)
    return sentences


def output_w2v_file(output_path, sentences):
    with open(output_path, 'w', encoding='utf8', newline='\n') as out:
        out.writelines(sentences)


if __name__ == '__main__':
    lines = load_training_data('data/labeledTrainData.tsv')
    sentences = make_w2v_data([line[2].strip('"') for line in lines])
    output_w2v_file('dataOutput/labeledTrainData_sentences.txt', sentences=sentences)
