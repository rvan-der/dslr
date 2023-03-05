import numpy as np
from TinyStatistician import TinyStatistician



def sigmoid(x):
	return 1 / (1 + np.exp(-x))



def shortenName(name):
	words = name.split()
	if len(words) == 1:
		return name[:7] + '.' if len(name) > 8 else name
	if len(name) > 8:
		return "".join([word[0] if word.lower() in ["the","of"] else word[0].upper() for word in words])
	return name



class DslrRobustScaler():

	ts = TinyStatistician()

	def __init__(self, data, percentiles=(25,75)):
		self.data = data
		self.numericFeatures = []
		for feature in data.columns:
			if any([not isinstance(value, float) for value in data[feature]]):
				continue
			self.numericFeatures.append(feature)
		self.medians = [self.ts.median(list(data[feature])) for feature in self.numericFeatures]
		self.ranges = [self.ts.percentile(list(data[feature]), max(percentiles)) -
						self.ts.percentile(list(data[feature]), min(percentiles))
						for feature in self.numericFeatures]


	def scale(self):
		scaledData = self.data.copy()
		for i,feature in enumerate(self.numericFeatures):
			scaledData[feature] = scaledData[feature].apply(lambda x:(x - self.medians[i]) / self.ranges[i])
		return scaledData


	def allScalingParams(self):
		model = {}
		for i,feature in enumerate(self.numericFeatures):
			model[feature] = {"median":self.medians[i], "range":self.ranges[i]}
		return model


	def scaleToModel(self, model):
		scaledData = self.data.copy()
		for feature in model["features"]["all"]:
			median = model["scaling"][feature]["median"]
			_range = model["scaling"][feature]["range"]
			scaledData[feature] = scaledData[feature].apply(lambda x:(x - median) / _range)
		return scaledData
