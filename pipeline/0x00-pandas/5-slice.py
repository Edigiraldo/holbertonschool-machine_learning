#!/usr/bin/env python3
"""Script to take data from 'High', 'Low', 'Close', 'Volume_(BTC)' colums of
dataset every 60 rows."""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]

print(df.tail())
