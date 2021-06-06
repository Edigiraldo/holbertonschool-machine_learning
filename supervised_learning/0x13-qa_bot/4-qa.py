#!/usr/bin/env python3
"""Function question_answer."""
qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Function hat answers questions from multiple reference texts.

    - corpus_path is the path to the corpus of reference documents.
    """
    inp = ""

    exit = ["exit", "quit", "goodbye", "bye"]
    while True:
        question = input("Q: ").lower()
        if question in exit:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        if len(reference) == 0:
            print("A: Sorry, I do not understand your question.")
            continue
        answ = qa(question, reference)
        if answ is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answ))
