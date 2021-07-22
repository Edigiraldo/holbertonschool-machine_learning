#!/usr/bin/env python3
"""Script to plot dataframe data."""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# YOUR CODE HERE
del df['Weighted_Price']
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df = df['2017-01-01':]

df['Close'].fillna(method="ffill", inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df_plot = pd.DataFrame()
df_plot['High'] = df['High'].groupby(pd.Grouper(freq='D')).max()
df_plot['Low'] = df['Low'].groupby(pd.Grouper(freq='D')).min()
df_plot['Open'] = df['Open'].groupby(pd.Grouper(freq='D')).mean()
df_plot['Close'] = df['Close'].groupby(pd.Grouper(freq='D')).mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].groupby(
                              pd.Grouper(freq='D')).sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].groupby(
                                   pd.Grouper(freq='D')).sum()

df_plot.plot()
plt.show()
# YOUR CODE HERE
