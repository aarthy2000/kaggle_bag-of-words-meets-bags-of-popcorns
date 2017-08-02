This is a repo for [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial) on [Kaggle](https://www.kaggle.com/)

# Experiment Flow

* Data preprocessing
* Generate [word2vec](https://code.google.com/archive/p/word2vec/) data
* Train a [CNN model](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)  for text classification
* Predict testing data


## Data preprocessing

* Observe data format
* Statistics
* Data cleaning
* word2vec

### Preprocessing steps of data cleaning

1. Remove the " character at the start/end of review
1. Replace \\" to "
1. Replace multiple \<br \/\> to space character
1. Tokenize the whole review (article)
1. Lemmatize all words

### Preprocessing steps of generating word2vec files

1. Remove the " character at the start/end of review
1. Replace \\" to "
1. Replace multiple \<br \/\> to space character
1. Split sentences from review (article)
1. Tokenize all sentences
1. Lemmatize all words


## Convolution Neural Network (CNN) Model

1. Preprocessing for CNN training data
1. Training model
1. Test model by validation set

### preprocessing

1. Split data into training set and validation set
1. Index all words in word2vec
1. Encode all the words of reviews in training data and validation data


