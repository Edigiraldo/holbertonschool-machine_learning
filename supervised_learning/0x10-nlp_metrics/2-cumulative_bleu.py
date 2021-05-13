#!/usr/bin/env python3
"""Function cumulative_bleu."""
import numpy as np


def build_ngram(words, n):
    """
    Function that builds the n-grams corresponding to the
    words.

    - words is a list of words.
    - n is an integer (> 0) indication the order of the
      n-gram (1-gram, 2-grams ...)

    Returns: a list with the n-grams.
    """
    n_grams = []
    for i in range(len(words) - n + 1):
        n_gram = words[i]
        for j in range(1, n):
            n_gram += ' ' + words[i + j]
        n_grams.append(n_gram)

    return n_grams


def ngram_bleu(references, sentence, n):
    """
    Function that calculates the n-gram BLEU score for a sentence.

    - references is a list of reference translations.
        - each reference translation is a list of the words in the
          translation.
    - sentence is a list containing the model proposed sentence.
    - n is the size of the n-gram to use for evaluation.

    Returns: the n-gram BLEU score.
    """
    refs_ngram = []
    for ref in references:
        n_grams = build_ngram(ref, n)
        refs_ngram.append(n_grams)

    sent_ngram = build_ngram(sentence, n)

    n_gram_count_sent = {}
    for n_gram in sent_ngram:
        n_gram_count_sent[n_gram] = n_gram_count_sent.get(n_gram, 0) + 1

    n_gram_count_clip = {}
    for ref in refs_ngram:
        for n_gram in set(ref):
            n_gram_count_clip[n_gram] = max(ref.count(n_gram),
                                            n_gram_count_clip.get(n_gram, 0))

    clipped_count = {}
    for ng in n_gram_count_sent.keys():
        clipped_count[ng] = min(n_gram_count_clip.get(ng, 0),
                                n_gram_count_sent[ng])

    Pn = sum(clipped_count.values()) / max(sum(n_gram_count_sent.values()), 1)

    return Pn


def cumulative_bleu(references, sentence, n):
    """
    Function that calculates the cumulative n-gram BLEU score for
    a sentence.

    - references is a list of reference translations.
        - each reference translation is a list of the words in the
          translation.
    - sentence is a list containing the model proposed sentence.
    - n is the size of the largest n-gram to use for evaluation.
    - All n-gram scores should be weighted evenly.

    Returns: the cumulative n-gram BLEU score.
    """
    Pns = []
    for i in range(1, n + 1):
        Pn = ngram_bleu(references, sentence, i)
        Pns.append(Pn)

    c = len(sentence)
    r = min([len(ref) - c for ref in references]) + c

    # Brevity penalty.
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)

    b = np.sum([np.log(Pns[i]) if Pns[i] != 0 else 0
                for i in range(n)]) / n

    cum_bleu = BP * np.exp(b)

    return cum_bleu
