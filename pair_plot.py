import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
import pandas as pd
from data_description import *
from data_processing import *


colors = {"Gryffindor": "#AE0001", "Slytherin": "#2A623D", "Ravenclaw": "#3D5CC3", "Hufflepuff": "#FFDB00"}

data = pd.read_csv('datasets/dataset_train.csv', index_col="Index").dropna()

numericFeatures = ["Arithmancy","Astronomy","Herbology"\
					,"Defense Against the Dark Arts","Divination"\
					,"Muggle Studies","Ancient Runes","History of Magic"\
					,"Transfiguration","Potions","Care of Magical Creatures"\
					,"Charms","Flying"]
scaler = DslrRobustScaler(data, numericFeatures, percentiles=(20,80))
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
		if x == y:
			axs[y][x].hist(scData.loc[scData["Hogwarts House"] == "Gryffindor", pltFeatures[x]], color=colors["Gryffindor"], alpha=0.5)
			axs[y][x].hist(scData.loc[scData["Hogwarts House"] == "Slytherin", pltFeatures[x]], color=colors["Slytherin"], alpha=0.5)
			axs[y][x].hist(scData.loc[scData["Hogwarts House"] == "Ravenclaw", pltFeatures[x]], color=colors["Ravenclaw"], alpha=0.5)
			axs[y][x].hist(scData.loc[scData["Hogwarts House"] == "Hufflepuff", pltFeatures[x]], color=colors["Hufflepuff"], alpha=0.5)
			if x:
				axs[y][x].sharey(axs[0][0])

		else:
			axs[y][x].scatter(scData[pltFeatures[x]], scData[pltFeatures[y]], marker='.', c=colorFeature, alpha=0.35)
		axs[y][x].set(xlabel=shortenName(pltFeatures[x]), ylabel=shortenName(pltFeatures[y]))
		axs[y][x].label_outer()

fig.tight_layout()
plt.show()