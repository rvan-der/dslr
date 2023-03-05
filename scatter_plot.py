#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from data_processing import *


colors = {"Gryffindor": "#AE0001", "Slytherin": "#2A623D", "Ravenclaw": "#3D5CC3", "Hufflepuff": "#FFDB00"}

try:
    data = pd.read_csv('datasets/dataset_train.csv', index_col="Index").dropna()
except:
    print("Error: could not read file")
    exit(1)

scaler = DslrRobustScaler(data)
scData = scaler.scale()
colorFeature = scData['Hogwarts House'].map(colors)

fig = plt.figure(figsize=(9,7), num="dslr")
plt.scatter(scData["Defense Against the Dark Arts"], scData["Astronomy"], marker='.', c=colorFeature, alpha=0.35)
plt.xlabel("Defense Against the Dark Arts")
plt.ylabel("Astronomy")
fig.suptitle("Similar features")
fig.tight_layout()


plt.savefig('scatter_plot.png')
plt.show()