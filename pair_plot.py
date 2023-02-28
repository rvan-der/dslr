import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
import pandas as pd
from data_description import *
from data_processing import *


data = pd.read_csv('datasets/dataset_train.csv', index_col="Index").dropna()
numericFeatures = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
scaler = DslrRobustScaler(data, numericFeatures, percentiles=(20,80))
scData = scaler.scale()
colors = scData['Hogwarts House'].map({"Gryffindor": "#AE0001", "Slytherin": "#2A623D", "Ravenclaw": "#222F5B", "Hufflepuff": "#FFDB00"})

pltFeatures = [ft for ft in data.columns if ft not in\
	["First Name","Last Name","Birthday","Best Hand"]]
nbFtr = len(pltFeatures)

fig = plt.figure(figsize=(16,12))
gspec = fig.add_gridspec(nbFtr, nbFtr, hspace=0, wspace=0)
axs = gspec.subplots()
for y in range(nbFtr):
	for x in range(nbFtr):
		if x == y:
			axs[y][x].hist(scData[pltFeatures[x]])
			if x:
				axs[y][x].sharey(axs[0][0])

		else:
			axs[y][x].scatter(scData[pltFeatures[x]], scData[pltFeatures[y]], marker='.', c=colors, alpha=0.35)
		axs[y][x].set(xlabel=shortenName(pltFeatures[x]), ylabel=shortenName(pltFeatures[y]))
		axs[y][x].label_outer()

fig.tight_layout()
plt.show()