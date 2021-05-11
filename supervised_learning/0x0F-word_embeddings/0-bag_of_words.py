#!/usr/bin/env python3
"""Function bag_of_words."""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Function that creates a bag of words embedding matrix.

    - sentences is a list of sentences to analyze.
    - vocab is a list of the vocabulary words to use for the
      analysis.

    Returns: embeddings, features.
        - embeddings is a numpy.ndarray of shape (s, f)
          containing the embeddings.
            - s is the number of sentences in sentences.
            - f is the number of features analyzed.
        - features is a list of the features used for embeddings.
    """
    CountVec = CountVectorizer(lowercase=True,
                               vocabulary=vocab)

    Count_data = CountVec.fit_transform(sentences)

    embeddings = Count_data.toarray()
    features = CountVec.get_feature_names()

    return embeddings, features
