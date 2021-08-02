#!/usr/bin/env python3
"""Function insert_school."""


def insert_school(mongo_collection, **kwargs):
    """
    Function that inserts a new document in a collection based on kwargs.

    - mongo_collection: collection.
    - kwargs: dict arguments.

    Return: new id of inserted document.
    """
    doc_id = mongo_collection.insert_one(kwargs).inserted_id

    return doc_id
