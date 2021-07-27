#!/usr/bin/env python3
"""Script to search for number of launches for each rocket."""
import requests


if __name__ == '__main__':
    response = requests.get('https://api.spacexdata.com/v4/launches')
    launches = response.json()
    rocket_ids_launches = {}

    for launch in launches:
        rocket_id = launch['rocket']
        if rocket_ids_launches.get(rocket_id, 0) == 0:
            rocket_ids_launches[rocket_id] = 1
        else:
            rocket_ids_launches[rocket_id] += 1

    rocket_launches = []
    for rocket_id in rocket_ids_launches.keys():
        rocket_url = 'https://api.spacexdata.com/v4/rockets/' + rocket_id
        rocket_response = requests.get(rocket_url)
        rocket_name = rocket_response.json()['name']
        num_launches = rocket_ids_launches[rocket_id]
        rocket_launches.append((rocket_name, num_launches))

    # Sort alphabetical
    rocket_launches.sort(key=lambda tup: tup[0])
    # Sort by launches
    rocket_launches.sort(key=lambda tup: tup[1], reverse=True)

    for rocket_name, num_launches in rocket_launches:
        print('{}: {}'.format(rocket_name, num_launches))
