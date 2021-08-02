#!/usr/bin/env python3
"""Function update_topics."""


def update_topics(mongo_collection, name, topics):
    """
    Function that changes all topics of a school document based on the name.

    - mongo_collection: pymongo collection object.
    - name: school name to update.
    - topics: list of topics approached in the school.
    """
    mongo_collection.update_many({"name": name},
                                 {"$set": {"topics": topics}})
