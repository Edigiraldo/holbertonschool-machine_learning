#!/usr/bin/env python3
"""Function answer_loop."""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Function that answers questions from a reference text.

    - reference is the reference text.
    """
    inp = ""

    exit = ["exit", "quit", "goodbye", "bye"]
    while True:
        question = input("Q: ").lower()
        if question in exit:
            print("A: Goodbye")
            break

        answ = question_answer(question, reference)
        if answ is None:
            print("Q: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answ))
