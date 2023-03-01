import sys
import json
import math
import padas as pd
import numpy as np
from data_processing import *


def prepare_data(data, modelFtrs, house):
	Y = [1 if h == house else 0 for h in data["Hogwarts House"]]
	X = []
	for i,student in data.iterrows():
		X.append([student[ft] for ft in modelFtrs] + [1])
	return X, Y


def sigmoid(x):
	return 1 / (1 + math.exp(x))


def logreg(X, Y, thetas, iters, lRate):
	m = len(Y)
	for i in range(iters):
		newThetas = []
		x,y = X[i],Y[i]
		for j in range(len(thetas)):
			newThetas.append(thetas[j] - lRate / m * sum((sigmoid(np.dot(x,thetas)) - y) * x[j]))
		thetas = newThetas
	return thetas




modelFtrs = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
nbModelFtrs = len(modelFtrs)
learningRate = 0.001
iterations = 100


if __name__ == "__main__":
	if sys.argc != 2:
		sys.exit(f"Got {sys.argc - 1} arguments instead of 1")
	fileName = sys.argv[1]

	try:
		rawData = pd.read_csv(fileName, index_col="Index").dropna()
	except Exception as e:
		sys.exit(f"An error occured while reading the dataset. {str(e)}")
	
	scaler = DslrRobustScaler(rawData, percentiles=(20,80))
	scaledData = scaler.scale()
	
	model = {"features": modelFtrs, "scaling": {}, "thetas": {}}
	for feature in modelFtrs:
		model["scaling"][feature] = scaler.featureScalingParams(feature)


	for house in ["Gryffindor","Slytherin","Ravenclaw","Hufflepuff"]:
		X,Y = prepare_data(scaledData, modelFtrs, house)
		model["thetas"][house] = logreg(X, Y, [0]*(nbModelFtrs+1), iterations, learningRate)

	try:
		with open("dslr_model.json", "w") as jsonFile:
			json.dump(model, jsonFile)
	except Exception as e:
		sys.exit(f"An error occured when saving the model to dslr_model.json: {str(e)}")


