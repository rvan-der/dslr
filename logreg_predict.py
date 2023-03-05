#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
from data_processing import *



def predict(x, thetas):
    ret = sigmoid(np.dot(x, thetas))
    return ret


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 2:
        sys.exit(f"Got {argc - 1} arguments instead of 1")
    fileName = sys.argv[1]

    try:
        data = pd.read_csv(fileName)
    except Exception as e:
        sys.exit(f"An error occured while reading the dataset. {str(e)}")

    try:
        with open("dslr_model.json", "r") as modelFile:
            model = json.load(modelFile)
    except Exception as e:
        sys.exit(f"An error occured while reading the model file. {str(e)}")

    for feature in model["features"]["all"]:
        missing = data[feature].isnull()
        data[feature] = [model["scaling"][feature]["median"] if missing[i] else x for i, x in enumerate(data[feature])]

    scaler = DslrRobustScaler(data, percentiles=(20,80))
    scaledData = scaler.scaleToModel(model)
    
    predictions = []

    for studId, student in scaledData.iterrows():
        pred = ("", -1)
        # print(studId)
        for house in ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]:
            x = [1] + list(student[model["features"][house]])
            probability = predict(x, model["thetas"][house])
            if probability > pred[1]:
                pred = (house, probability)
            # print(house,probability)
        predictions.append(pred[0])

    scaledData["prediction"] = predictions
    scaledData.to_csv("houses.csv", columns=["Index","prediction"], header=["Index", "Hogwarts House"], index=False)



