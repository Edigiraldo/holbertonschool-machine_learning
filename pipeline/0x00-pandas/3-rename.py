#!/usr/bin/env python3
"""Renames colum Timestamp to Datetime, converts colum to Datetime and
takes just Datetime and Close columns from dataframe."""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
df = df[['Datetime', 'Close']]
# YOUR CODE HERE

print(df.tail())
