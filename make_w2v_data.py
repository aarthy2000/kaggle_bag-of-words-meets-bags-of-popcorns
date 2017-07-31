import re
import nltk
from nltk.stem import WordNetLemmatizer
from corpus_loader import *
nltk.data.path.append('nltk_data/')


def make_w2v_data(reviews):
    sentences = list()
    for review in reviews:
        review = re.sub('(<br \/>)+', ' ', review)  # replace <br /> to space
        review = review.replace('\\"', '"')
        review_sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', review)  # split to sentences
        # append \n and remove space
        review_sentences = [(sentence + '\n').strip(' ') for sentence in review_sentences]
        sentences.extend(review_sentences)

    sentences = sentences_preprocessing(sentences=sentences)
    return sentences


def sentences_preprocessing(sentences):
    lemmatizer = WordNetLemmatizer()

    token_lemma_sentences = list()
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        token_lemma_sentences.append(' '.join(tokens) + '\n')

    return token_lemma_sentences


def output_w2v_file(output_path, sentences):
    with open(output_path, 'w', encoding='utf8', newline='\n') as out:
        out.writelines(sentences)


if __name__ == '__main__':
    lines = load_training_data('data/labeledTrainData.tsv')
    sentences = make_w2v_data([line[2].strip('"') for line in lines])
    output_w2v_file('dataOutput/labeledTrainData_sentences.txt', sentences=sentences)
