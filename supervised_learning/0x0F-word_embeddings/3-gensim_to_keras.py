#!/usr/bin/env python3
"""Function gensim_to_keras."""
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """
    Function that converts a gensim word2vec model to a keras
    Embedding layer.

    - model is a trained gensim word2vec models.

    Returns: the trainable keras Embedding.
    """
    keras_emb = model.wv.get_keras_embedding(True)

    return keras_emb
