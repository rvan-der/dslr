import sys
import json
import pandas as pd
import numpy as np
from data_processing import *


def prepare_data(data, modelFtrs, house):
	Y = [1 if h == house else 0 for h in data["Hogwarts House"]]
	X = []
	for i,student in data.iterrows():
		X.append([student[ft] for ft in modelFtrs] + [1])
	return X, Y


def logreg(X, Y, thetas, iters, lRate):
	m = len(Y)
	for it in range(iters):
		newThetas = []
		for j in range(len(thetas)):
			gradient = 1 / m * sum((sigmoid(np.dot(X[i],thetas)) - Y[i]) * X[i][j] for i in range(m))
			newThetas.append(thetas[j] - lRate * gradient)
		thetas = newThetas
	return thetas




modelFtrs = ["Astronomy","Herbology","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Charms","Flying"]
nbModelFtrs = len(modelFtrs)
learningRate = 0.001
iterations = 100


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit(f"Got {sys.argc - 1} arguments instead of 1")
	fileName = sys.argv[1]

	try:
		data = pd.read_csv(fileName, index_col="Index").dropna()
	except Exception as e:
		sys.exit(f"An error occured while reading the dataset. {str(e)}")
	
	scaler = DslrRobustScaler(data, percentiles=(20,80))
	scaledData = scaler.scale()
	
	model = {"features": modelFtrs, "scaling": {}, "thetas": {}}
	for feature in modelFtrs:
		model["scaling"][feature] = scaler.featureScalingParams(feature)

	for house in ["Gryffindor","Slytherin","Ravenclaw","Hufflepuff"]:
		X,Y = prepare_data(scaledData, modelFtrs, house)
		model["thetas"][house] = logreg(X, Y, [0]*(nbModelFtrs+1), iterations, learningRate)

	try:
		with open("dslr_model.json", "w") as jsonFile:
			json.dump(model, jsonFile, indent=4)
	except Exception as e:
		sys.exit(f"An error occured when saving the model to dslr_model.json: {str(e)}")


