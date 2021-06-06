#!/usr/bin/env python3
"""semantic_search function."""
import tensorflow as tf
import tensorflow_hub as hub
import os


def semantic_search(corpus_path, sentence):
    """
    Function that performs semantic search on a corpus of documents.

    - corpus_path is the path to the corpus of reference documents on which
      to perform semantic search.
    - sentence is the sentence from which to perform semantic search.

    Returns: the reference text of the document most similar to sentence.
    """
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    sen_embed = embed([sentence])

    max_sim = -2
    most_similar = ''

    ref_files = os.listdir(corpus_path)
    ref_files = [x for x in ref_files if x.endswith('.md')]

    for ref_file in ref_files:
        with open(corpus_path + "/" + ref_file) as f:
            f_read = f.read()
            ref_embed = embed([f_read])
        similarity = tf.tensordot(sen_embed, ref_embed, axes=[[1], [1]])
        if max_sim < similarity:
            max_sim = similarity
            most_similar = f_read

    return most_similar
