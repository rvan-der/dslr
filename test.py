import json
import pandas as pd
import numpy as np
from data_processing import *

with open("dslr_model.json") as modelFile:
    model = json.load(modelFile)
train = DslrRobustScaler(pd.read_csv("datasets/dataset_train.csv").dropna()).scale()
test = DslrRobustScaler(pd.read_csv("datasets/dataset_test.csv")).scaleToModel(model)
ftrs = ["Astronomy","Herbology","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Charms","Flying"]
print("TRAIN")
print(train[ftrs].describe())
print("TEST")
print(test[ftrs].describe())
