This is a repo for [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial) on [Kaggle](https://www.kaggle.com/)

# Experiment Flow

* Data pre-processing
* Generate [word2vec](https://code.google.com/archive/p/word2vec/) data
* Train a [CNN model](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)  for text classification
* Predict testing data


## Data pre-processing

* observe data format
* statistics
* data cleaning
* word2vec

### Pre-processing steps of data cleaning

1. Remove the " character at the start/end of review
1. Replace \\" to "
1. Replace multiple \<br \/\> to space character
1. Tokenize the whole review (article)
1. Lemmatize all words

### Steps of generating word2vec files

1. Remove the " character at the start/end of review
1. Replace \\" to "
1. Replace multiple \<br \/\> to space character
1. Split sentences from review (article)
1. Tokenize all sentences
1. Lemmatize all words
