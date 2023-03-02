import json
import pandas as pd
import numpy as np
from data_processing import *



def predict(x, thetas):
    return sigmoid(np.dot(x, thetas))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Got {sys.argc - 1} arguments instead of 1")
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

    scaler = DslrRobustScaler(data, percentiles=(20,80))
    scaledData = scaler.scaleToModel(model)
    predictions = []

    for _,student in scaledData.iterrows():
        pred = ("", -1)
        for house in ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]:
            x = list(student[model["features"]]) + [1]
            prob = predict(x, model["thetas"][house])
            if prob > pred[1]:
                pred = (house, prob)
        predictions.append(pred[0])

    scaledData["prediction"] = predictions
    scaledData.to_csv("houses.csv", columns=["Index","prediction"], header=["Index", "Hogwarts House"], index=False)



