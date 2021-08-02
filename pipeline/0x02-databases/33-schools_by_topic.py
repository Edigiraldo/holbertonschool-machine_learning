#!/usr/bin/env python3
"""Function schools_by_topic"""


def schools_by_topic(mongo_collection, topic):
    """
    function that returns the list of school having a specific topic.

    - mongo_collection: pymongo collection object.
    - topics: topic searched.

    Return: list of school having a specific topic.
    """
    l_results = []
    results = mongo_collection.find({"topics": {"$all": [topic]}})
    for result in results:
        l_results.append(result)

    return l_results
