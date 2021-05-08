#!/usr/bin/env python3
"""Function preprocess.
csv_path = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
"""
# import pandas as pd


def preprocess_data(csv_path):
    """
    Function that preprocess data in csv_path path. It takes
    hourly data, splits it in train, val and test dataframes
    and normalizes it.

    Return: train_df, val_df, test_df. Training data, validation
    data and test data in a ratio of 0.7, 0.2, 0.1 respectively.
    """
    df = pd.read_csv(csv_path)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True, drop=True)
    df = df.drop(['Timestamp'], axis=1)

    # take hourly data.
    df = df[::60]

    # Splitting data -> train, val, test.
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    # Normalize all data.
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
