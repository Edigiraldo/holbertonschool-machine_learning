#!/usr/bin/env python3
"""Script to fetch spaceX API for next launch."""
import requests
import time


if __name__ == '__main__':
    current_time = time.time()

    response = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    launches = response.json()

    soonest_launch_idx = 0
    soonest_launch_time = launches[0]['date_unix']
    for i in range(len(launches)):
        launch_time = launches[i]['date_unix']

        if launch_time > current_time and launch_time < soonest_launch_time:
            soonest_launch_idx = i
            soonest_launch_time = launch_time

    soonest_launch = launches[soonest_launch_idx]

    launch_name = soonest_launch['name']
    local_time_date = soonest_launch['date_local']
    rocket_id = soonest_launch['rocket']
    launchpad_id = soonest_launch['launchpad']

    rocket_url = f'https://api.spacexdata.com/v4/rockets/{rocket_id}'
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()['name']

    launchpad_url = f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}'
    launchpad_response = requests.get(launchpad_url)
    launchpad_name = launchpad_response.json()['name']
    launchpad_locality = launchpad_response.json()['locality']

    print(f'{launch_name} ({local_time_date}) {rocket_name} -' +
          f' {launchpad_name} ({launchpad_locality})')
