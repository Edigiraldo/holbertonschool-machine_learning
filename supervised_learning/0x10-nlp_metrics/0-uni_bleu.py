#!/usr/bin/env python3
"""Function uni_bleu."""
import numpy as np


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram BLEU score for a sentence.

    - references is a list of reference translations.
        - each reference translation is a list of the words in the
          translation.
    - sentence is a list containing the model proposed sentence.

    Returns: the unigram BLEU score.
    """

    word_count_sent = {}
    for word in sentence:
        word_count_sent[word] = word_count_sent.get(word, 0) + 1

    word_count_clip = {}
    for ref in references:
        for word in set(ref):
            word_count_clip[word] = max(ref.count(word),
                                        word_count_clip.get(word, 0))

    clipped_count = {}
    for w in word_count_sent.keys():
        clipped_count[w] = min(word_count_clip.get(w, 0),
                               word_count_sent[w])

    P1 = sum(clipped_count.values()) / max(sum(word_count_sent.values()), 1)

    c = len(sentence)
    r = min([len(ref) - c for ref in references]) + c

    # Brevity penalty.
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)

    BLEU_unigram = BP * P1

    return BLEU_unigram
