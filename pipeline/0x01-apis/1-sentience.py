#!/usr/bin/env python3
"""Function sentientPlanets."""
import requests


def sentientPlanets():
    """
    Function that returns the list of names of the home planets
    of all sentient species.
    """
    checked_planet_urls = {}
    sentient_planets = []

    next_page_url = 'https://swapi-api.hbtn.io/api/species/'
    r = requests.get(next_page_url)
    r_json = r.json()

    while next_page_url:
        r = requests.get(next_page_url)
        r_json = r.json()
        next_page_url = r_json['next']

        species = r_json['results']
        for s in species:
            designation = s.get('designation', None)
            classification = s.get('classification', None)
            if designation == 'sentient' or classification == 'sentient':
                planet_url = s['homeworld']
                if (checked_planet_urls.get(planet_url, False)
                   or planet_url is None):
                    continue
                else:
                    planet = requests.get(planet_url).json()['name']
                    sentient_planets.append(planet)

                    checked_planet_urls[planet_url] = True
    sentient_planets.sort()

    return sentient_planets


"""
# Uncomment here to check if intranet output is same as here.
planets_intranet = ['Endor',
'Naboo','Coruscant','Kamino','Geonosis',
'Utapau','Kashyyyk','Cato Neimoidia','Rodia',
'Nal Hutta','unknown','Trandosha','Mon Cala',
'Sullust','Toydaria','Malastare','Ryloth',
'Aleen Minor','Vulpter','Troiken','Tund',
'Cerea','Glee Anselm','Iridonia','Tholoth',
'Iktotch','Quermia','Dorin','Champala',
'Mirial','Zolan','Ojom','Skako',
'Muunilinst','Shili','Kalee']
planets_intranet.sort()

sentient_planets = sentientPlanets()
print(len(sentient_planets), '	<--->	',  len(planets_intranet))

for i in range(36):
    print(sentient_planets[i], '  <--->   ', planets_intranet[i])
"""
