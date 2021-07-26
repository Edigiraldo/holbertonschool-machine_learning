#!/usr/bin/env python3
import requests
from sys import argv
import time
"""Script to request Github API for users location."""


if __name__ == '__main__':
    user_address = argv[-1]

    params = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(user_address, params=params)
    status_code = response.status_code

    if status_code == 200:
        json = response.json()
        location = json['location']
        print(location)

    elif status_code == 403:
        ratelimit = int(response.headers['X-Ratelimit-Reset'])
        time_now = time.time()
        minutes = int((ratelimit - time_now) / 60)

        print(f"Reset in {minutes} min")

    elif status_code == 404:
        print("Not found")
