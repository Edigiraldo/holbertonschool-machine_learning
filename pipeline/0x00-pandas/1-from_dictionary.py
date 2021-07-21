#!/usr/bin/env python3
"""Script to create a pd.DataFrame from a dictionary."""
import pandas as pd


data = {'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']}
idxs = ['A', 'B', 'C', 'D']

df = pd.DataFrame(data, index=idxs)
