# from data_description import *
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("datasets/dataset_train.csv", index_col="Index")

charms = list(data["Charms"])
astro = list(data["Astronomy"])

fig,axs=plt.subplots(2,2,figsize=(2,2))
axs[0][0].hist(charms)
axs[0][0].set(xlabel="charms",ylabel="charms")

axs[0][1].scatter(astro, charms,marker='+',c=data["Hogwarts House"].astype('category').cat.codes)
axs[0][1].set(xlabel="astro", ylabel="charms")

axs[1][1].hist(astro)
axs[1][1].set(xlabel="astro",ylabel="astro")

axs[1][0].scatter(charms, astro,marker='+',c=data["Hogwarts House"].astype('category').cat.codes)
axs[1][0].set(xlabel="charms", ylabel="astro")

for ax in axs.flat:
    ax.label_outer()

plt.show()