import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
import pandas as pd
from data_description import *
from data_processing import *


data = pd.read_csv('datasets/dataset_train.csv', index_col="Index")\
		.dropna()\
		.drop(columns=["Hogwarts House","First Name","Last Name","Birthday","Best Hand"])

numericFeatures = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
scaler = DslrRobustScaler(data, numericFeatures, percentiles=(20,80))
scdata = scaler.scale()
features = list(data.columns)
nbFtr = len(features)
fig, axs = plt.subplots(nbFtr, nbFtr, figsize=(16,13))
for y in range(nbFtr):
	for x in range(nbFtr):
		axs[y][x].scatter(list(scdata[features[x]]), list(scdata[features[y]]))

mplrc("xtick", labelsize=1)
mplrc("ytick", labelsize=1)
plt.show()