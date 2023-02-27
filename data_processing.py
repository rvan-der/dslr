from data_description import *


class DslrRobustScaler():

	def __init__(self, data, numericFeatures, percentiles=(25,75)):
		self.data = data
		self.numericFeatures = numericFeatures
		self.percentiles = percentiles
		self.medians = [median(data[feature]) for feature in numericFeatures]
		self.ranges = [quantile(max(percentiles),100,data[feature]) - \
		quantile(min(percentiles),100,data[feature]) for feature in numericFeatures]


	def featureScalingParams(self, feature):
		featureIndex = self.numericFeatures.index(feature)
		return {"median":self.medians[featureIndex], "range":self.ranges[featureIndex]}


	def scale(self):
		scaledData = self.data.copy()
		for i,feature in enumerate(self.numericFeatures):
			scaledData[feature] = scaledData[feature].apply(lambda x:(x - self.medians[i]) / self.ranges[i])
		return scaledData