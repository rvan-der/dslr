#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from data_description import *
from data_processing import *


colors = {"Gryffindor": "#AE0001", "Slytherin": "#2A623D", "Ravenclaw": "#3D5CC3", "Hufflepuff": "#FFDB00"}

try:
    data = pd.read_csv('datasets/dataset_train.csv', index_col="Index").dropna()
except:
    print("Error: could not read file")
    exit(1)

scaler = DslrRobustScaler(data, percentiles=(20,80))
scData = scaler.scale()
colorFeature = scData['Hogwarts House'].map(colors)

pltFeatures = [ft for ft in data.columns if ft not in\
    ["First Name","Last Name","Birthday","Best Hand", "Hogwarts House"]]
nbFtr = len(pltFeatures)

fig = plt.figure(figsize=(16,12))
gspec = fig.add_gridspec(nbFtr, nbFtr, hspace=0, wspace=0)
axs = gspec.subplots()
for y in range(nbFtr):
    for x in range(nbFtr):
        if x < y:
            axs[y][x].scatter(scData[pltFeatures[x]], scData[pltFeatures[y]], marker='.', c=colorFeature, alpha=0.35)
            axs[y][x].set(xlabel=shortenName(pltFeatures[x]), ylabel=shortenName(pltFeatures[y]))
            axs[y][x].label_outer()
        else:
            axs[y][x].set_axis_off()

fig.tight_layout()
plt.savefig('scatter_plot.png')
plt.show()