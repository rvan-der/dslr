#!/usr/bin/env python3

import sys
import time
import json
import pandas as pd
import numpy as np
from threading import Thread
from data_processing import *



class LogregThread(Thread):
	def __init__(self, X, Y, iters, lRate, house, model, progress):
		Thread.__init__(self)
		self.X = X
		self.Y = Y
		self.iters = iters
		self.lRate = lRate
		self.house = house
		self.model = model
		self.progress = progress

	def run(self):
		m = len(self.Y)
		for it in range(self.iters):
			thetas = self.model["thetas"][self.house]
			newThetas = []
			for j in range(len(thetas)):
				gradient = 1 / m * sum((sigmoid(np.dot(self.X[i],thetas)) - self.Y[i]) * self.X[i][j] for i in range(m))
				newThetas.append(thetas[j] - self.lRate * gradient)
			self.model["thetas"][self.house] = newThetas
			self.progress[self.house] += 1



def prepare_data(data, modelFtrs, house):
	Y = [1 if h == house else 0 for h in data["Hogwarts House"]]
	X = []
	for i,student in data.iterrows():
		X.append([1] + [student[ft] for ft in modelFtrs])
	return X, Y



if __name__ == "__main__":
	
	learningRate = 0.1
	iterations = 200
	houses = ["Gryffindor","Slytherin","Ravenclaw","Hufflepuff"]

	argc = len(sys.argv)
	if argc != 2:
		sys.exit(f"Got {argc - 1} arguments instead of 1")
	fileName = sys.argv[1]

	try:
		data = pd.read_csv(fileName, index_col="Index").dropna()
	except Exception as e:
		sys.exit(f"An error occured while reading the dataset. {str(e)}")
	
	scaler = DslrRobustScaler(data)
	scaledData = scaler.scale()

	ftrsPerHouse = {
		"Gryffindor": ["Astronomy", "Ancient Runes", "Herbology", "Charms", "Flying", "Transfiguration", "History of Magic"],
		"Slytherin": ["Astronomy", "Ancient Runes", "Herbology", "Charms", "Divination"],
		"Ravenclaw": ["Astronomy", "Ancient Runes", "Herbology", "Charms", "Muggle Studies"],
		"Hufflepuff": ["Astronomy", "Ancient Runes", "Herbology", "Charms"]
	}
	lst = ftrsPerHouse["Gryffindor"] + ftrsPerHouse["Slytherin"] + ftrsPerHouse["Ravenclaw"] + ftrsPerHouse["Hufflepuff"]
	allFeatures = list(set(lst))
	ftrsPerHouse["all"] = allFeatures
	
	# Initialise the model and threads progress.
	model = {"features": ftrsPerHouse, "scaling": {}, "thetas": {}}
	progress = {}
	for house in houses:
		model["thetas"][house] = [0] * (len(ftrsPerHouse[house]) + 1)
		progress[house] = 0
	model["scaling"] = scaler.allScalingParams()
	
	# Prepare the threads
	print("Loading threads...")
	threads = []
	for house in houses:
		X,Y = prepare_data(scaledData, ftrsPerHouse[house], house)
		th = LogregThread(X, Y, iterations, learningRate, house, model, progress)
		threads.append(th)

	# Launch threads:
	for th in threads:
		th.start()

	# Progress bar
	print("Training...")
	iterTotal = 4 * iterations
	width = 50
	while any([th.is_alive() for th in threads]):
		progTotal = sum(progress[h] for h in houses)
		full = int(progTotal / iterTotal * width)
		empty = width - full
		sys.stdout.write(f"[{'='*full + ' '*empty}]")
		sys.stdout.flush()
		sys.stdout.write("\r")
		time.sleep(1)
	else:
		print(f"[{'='*width}]\nTraining done !")

	try:
		with open("dslr_model.json", "w") as jsonFile:
			json.dump(model, jsonFile, indent=4)
	except Exception as e:
		sys.exit(f"An error occured when saving the model to 'dslr_model.json': {str(e)}")


