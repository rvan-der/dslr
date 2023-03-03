import numpy as np
from data_description import *



def sigmoid(x):
	return 1 / (1 + np.exp(x))



class DslrRobustScaler():

	def __init__(self, data, percentiles=(25,75)):
		self.data = data
		self.numericFeatures = []
		for feature in data.columns:
			if any([not isinstance(value, float) for value in data[feature]]):
				continue
			self.numericFeatures.append(feature)
		self.percentiles = percentiles
		self.medians = [median(data[feature]) for feature in self.numericFeatures]
		self.ranges = [quantile(max(percentiles),100,data[feature]) - \
		quantile(min(percentiles),100,data[feature]) for feature in self.numericFeatures]


	def featureScalingParams(self, feature):
		featureIndex = self.numericFeatures.index(feature)
		return {"median":self.medians[featureIndex], "range":self.ranges[featureIndex]}


	def allScalingParams(self):
		model = {}
		for i,feature in enumerate(self.numericFeatures):
			model[feature] = {"median":self.medians[i], "range":self.ranges[i]}
		return model


	def scale(self):
		scaledData = self.data.copy()
		for i,feature in enumerate(self.numericFeatures):
			scaledData[feature] = scaledData[feature].apply(lambda x:(x - self.medians[i]) / self.ranges[i])
		return scaledData


	def scaleToModel(self, model):
		scaledData = self.data.copy()
		for feature in model["features"]["all"]:
			median = model["scaling"][feature]["median"]
			_range = model["scaling"][feature]["range"]
			scaledData[feature] = scaledData[feature].apply(lambda x:(x - median) / _range)
		return scaledData
