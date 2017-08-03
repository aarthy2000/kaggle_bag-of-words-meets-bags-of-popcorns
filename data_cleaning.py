import re
import nltk
from nltk.stem import WordNetLemmatizer
from corpus_loader import load_training_data
nltk.data.path.append('nltk_data/')


def clean_training_data(lines):
    lemmatizer = WordNetLemmatizer()

    output_lines = list()
    for id, sentiment, review in lines:
        print(id)
        review = review.strip('"')
        review = re.sub('(<br \/>)+', ' ', review)  # replace <br /> to space
        review = review.replace('\\"', '"')
        tokens = nltk.word_tokenize(review)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        output_lines.append(id + '\t' + sentiment + '\t' + ' '.join(tokens) + '\n')
    return output_lines


def clean_training_data(lines):
    lemmatizer = WordNetLemmatizer()

    output_lines = list()
    for id, review in lines:
        print(id)
        review = review.strip('"')
        review = re.sub('(<br \/>)+', ' ', review)  # replace <br /> to space
        review = review.replace('\\"', '"')
        tokens = nltk.word_tokenize(review)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        output_lines.append(id + '\t' + ' '.join(tokens) + '\n')
    return output_lines


def generate_w2v_data(reviews):
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


def output_file(output_path, lines):
    with open(output_path, 'w', encoding='utf8', newline='\n') as out:
        out.writelines(lines)


if __name__ == '__main__':
    # # clean training data
    # lines = load_training_data('data/labeledTrainData.tsv')
    # output_lines = clean_training_data(lines)
    # output_file(output_path='dataOutput/labeledTrainData_clean.tsv', lines=output_lines)

    # # clean testing data
    # lines = load_testing_data('data/testData.tsv')
    # output_lines = clean_training_data(lines)
    # output_file(output_path='dataOutput/testData_clean.tsv', lines=output_lines)

    # generate w2v training file
    lines = load_training_data('data/labeledTrainData.tsv')
    sentences = generate_w2v_data([line[2].strip('"') for line in lines])
    output_file('dataOutput/labeledTrainData_sentences.txt', lines=sentences)
