#!/usr/bin/python3


import pandas as pd
import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(1, path)
from TinyStatistician import TinyStatistician as ts


# check args
if len(sys.argv) != 2:
    print("Usage: python3 describe.py dataset.csv")
    exit(1)

# read csv
try:
    df = pd.read_csv(sys.argv[1])
except:
    print("Error: could not read file")
    exit(1)

# check if empty
if df.empty:
    print("Error: empty file")
    exit(1)

# global parameters
stats = ts()
numeric_features = ['Arithmancy',
                    'Astronomy',
                    'Herbology',
                    'Defense Against the Dark Arts',
                    'Divination',
                    'Muggle Studies',
                    'Ancient Runes',
                    'History of Magic',
                    'Transfiguration',
                    'Potions',
                    'Care of Magical Creatures',
                    'Charms',
                    'Flying']


# describe numeric features

try:
    numeric_features_df = df[numeric_features]
except:
    print("Error: could not read features. Check that all features are present in the dataset.")
    exit(1)

description = {}
for feature in numeric_features_df.columns:
    data = np.array(pd.to_numeric(
        numeric_features_df[feature], errors='coerce').dropna().astype(float))

    # if we need to drop negative values
    # data = data[data >= 0]

    description[feature] = {}
    for i in data:
        if not isinstance(i, (int, float)):
            print(i)
            print("Error: invalid data")
            exit(1)
    description[feature]['Count'] = len(data)
    description[feature]['Mean'] = stats.mean(data)
    description[feature]['Std'] = stats.std(data)
    description[feature]['Min'] = stats.min(data)
    description[feature]['25%'] = stats.percentile(data, 25)
    description[feature]['50%'] = stats.percentile(data, 50)
    description[feature]['75%'] = stats.percentile(data, 75)
    description[feature]['Max'] = stats.max(data)

result = pd.DataFrame(description)
print(result)