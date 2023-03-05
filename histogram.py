#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
import pandas as pd
import numpy as np
from data_processing import *

colors = {"Gryffindor": "#AE0001", "Slytherin": "#2A623D",
          "Ravenclaw": "#3D5CC3", "Hufflepuff": "#FFDB00"}

try:
    data = pd.read_csv('datasets/dataset_train.csv', index_col="Index").dropna()
except:
    print("Error: could not read file")
    exit(1)

scaler = DslrRobustScaler(data)
scData = scaler.scale()
colorFeature = scData['Hogwarts House'].map(colors)

pltFeatures = [ft for ft in data.columns if ft not in
               ["First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]]
nbFtr = len(pltFeatures)

fig = plt.figure(figsize=(6, 12), num="dslr")
gspec = fig.add_gridspec(nbFtr, 1, hspace=0, wspace=0)
axs = gspec.subplots()
for x in range(nbFtr):
    axs[x].hist(scData.loc[scData["Hogwarts House"] == "Gryffindor",
                           pltFeatures[x]], color=colors["Gryffindor"], alpha=0.5)
    axs[x].hist(scData.loc[scData["Hogwarts House"] == "Slytherin",
                           pltFeatures[x]], color=colors["Slytherin"], alpha=0.5)
    axs[x].hist(scData.loc[scData["Hogwarts House"] == "Ravenclaw",
                           pltFeatures[x]], color=colors["Ravenclaw"], alpha=0.5)
    axs[x].hist(scData.loc[scData["Hogwarts House"] == "Hufflepuff",
                           pltFeatures[x]], color=colors["Hufflepuff"], alpha=0.5)
    axs[x].set_xlim(-4, 4)
    axs[x].sharex(axs[0])
    axs[x].text(2, 25, pltFeatures[x], fontsize=8)
    if x != nbFtr - 1:
        axs[x].set_xticklabels([])


fig.suptitle("Score distributions")
fig.tight_layout()
plt.legend(["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"],
           loc='lower left')
plt.savefig("histograms.png")
plt.show()