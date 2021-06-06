#!/usr/bin/env python3
"""question_answer function."""


def question_answer(question, reference):
    """
    Function that finds a snippet of text within a reference document to answer
    a question.

    - question is a string containing the question to answer.
    - reference is a string containing the reference document from which to
      find the answer.

    Returns: a string containing the answer.
    """
    mod = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(mod)
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # tokenize data
    question_tokens = tokenizer.tokenize(question)
    ref_tokens = tokenizer.tokenize(reference)

    # Formatting input for BERT
    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] + ref_tokens
              + ['[SEP]'])

    # To ids and create masks.
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    attent_mask = [1] * len(input_word_ids)
    token_type_ids = ([0] * (1 + len(question_tokens) + 1)
                      + [1] * (len(ref_tokens) + 1))

    # To tensors and create new axis for batch size.
    input_word_ids, attent_mask, token_type_ids = map(lambda t: tf.expand_dims(
      tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids,
                                                    attent_mask,
                                                    token_type_ids))

    # BERT model.
    outputs = model([input_word_ids, attent_mask, token_type_ids])

    # start and end of answer in reference file.
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    # Get answer and return it.
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if len(answer) <= 1:
        return None

    return answer
