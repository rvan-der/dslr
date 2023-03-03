import sys
import json
import pandas as pd
import numpy as np
from data_processing import *


def prepare_data(data, modelFtrs, house):
	Y = [1 if h == house else 0 for h in data["Hogwarts House"]]
	X = []
	for i,student in data.iterrows():
		X.append([1] + [student[ft] for ft in modelFtrs])
	return X, Y



def logreg(X, Y, thetas, iters, lRate):
	print("plop")
	m = len(Y)
	for it in range(iters):
		newThetas = []
		grads = []
		for j in range(len(thetas)):
			gradient = 1 / m * sum((sigmoid(np.dot(X[i],thetas)) - Y[i]) * X[i][j] for i in range(m))
			grads.append(gradient)
			newThetas.append(thetas[j] - lRate * gradient)
		thetas = newThetas
		print(grads)
	return thetas



if __name__ == "__main__":

	argc = len(sys.argv)
	if argc != 2:
		sys.exit(f"Got {argc - 1} arguments instead of 1")
	fileName = sys.argv[1]

	try:
		data = pd.read_csv(fileName, index_col="Index").dropna()
	except Exception as e:
		sys.exit(f"An error occured while reading the dataset. {str(e)}")

	f = ["Astronomy", "Ancient Runes", "Herbology", "Charms"]
	ftrsPerHouse = {
		"Gryffindor": f,
		"Slytherin": f + ["Divination"],
		"Ravenclaw": f,
		"Hufflepuff": f
	}
	l = ftrsPerHouse["Gryffindor"] + ftrsPerHouse["Slytherin"] + ftrsPerHouse["Ravenclaw"] + ftrsPerHouse["Hufflepuff"]
	allFeatures = list(set(l))
	ftrsPerHouse["all"] = allFeatures
	learningRate = 0.1
	iterations = 500
	
	scaler = DslrRobustScaler(data, percentiles=(20,80))
	scaledData = scaler.scale()
	
	model = {"features": ftrsPerHouse, "scaling": {}, "thetas": {}}

	model["scaling"] = scaler.allScalingParams()

	for house in ["Gryffindor","Slytherin","Ravenclaw","Hufflepuff"]:
		X,Y = prepare_data(scaledData, ftrsPerHouse[house], house)
		nbFtrs = len(ftrsPerHouse[house])
		model["thetas"][house] = logreg(X, Y, [0]*(nbFtrs+1), iterations, learningRate)

	try:
		with open("dslr_model.json", "w") as jsonFile:
			json.dump(model, jsonFile, indent=4)
	except Exception as e:
		sys.exit(f"An error occured when saving the model to 'dslr_model.json': {str(e)}")


