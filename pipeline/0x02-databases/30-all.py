#!/usr/bin/env python3
"""Functin list_all."""


def list_all(mongo_collection):
    """
    Function that lists all documents in a collection.

    - mongo_collection: mongo collection.

    Return: lists of documents in collection.
    """
    docs = []

    cursor = mongo_collection.find({})
    for document in cursor:
        docs.append(document)

    return docs
