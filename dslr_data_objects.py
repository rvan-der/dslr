import csv
from dslr_exceptions import *



class Feature():

	def __init__(self, name, values, _type):
		self.mean = None
		self.lowerQuartile = None
		self.median = None
		self.upperQuartile = None
		self.IQR = None
		self.stdDeviation = None
		self.name = name
		self.shortName = None
		self.values = values
		self.count = len(values)
		self._type = _type
		if _type == "numeric":
			self.values.sort()


	def getShortName(self):
		if self.shortName != None:
			return self.shortName
		words = self.name.split()
		if len(words) == 1:
			self.shortName = self.name
			if len(self.name) > 8:
				self.shortName = self.name[:7] + '.'
		else:
			if len(self.name) > 8:
				self.shortName = ".".join([word[0].upper() for word in words if word.lower() not in ["the","a","of","from"]])
		return self.shortName

			

	def quantile(self, k, q):
		if self._type != "numeric":
			raise DataError("Can't compute a quantile for non numeric features.")
		if q < 2 or q % 1:
			raise DataError(f"Invalid q value ({q})")
		if k < 1 or k >= q:
			raise DataError(f"Invalid k value ({k}) for q={q}")
		if self.count < q:
			raise DataError(f"Insufficient values to compute {q}-quantile (count={count})")
		index = self.count * k / q
		if index % 1:
			return self.values[int(index)]
		index = int(index)
		return (self.values[index - 1] + self.values[index]) / 2


	def getMean(self):
		if self.mean == None:
			if self._type != "numeric":
				raise DataError("Can't compute a mean for non numeric features.")
			self.mean = sum(self.values) / self.count
		return self.mean
		

	def getLowerQuartile(self):
		if self.lowerQuartile == None:
			self.lowerQuartile = self.quantile(1, 4)
		return self.lowerQuartile


	def getUpperQuartile(self):
		if self.upperQuartile == None:
			self.upperQuartile = self.quantile(3, 4)
		return self.upperQuartile


	def getMedian(self):
		if self.median == None:
			self.median = self.quantile(1,2)
		return self.median


	def getIQR(self):
		return self.getUpperQuartile() - self.getLowerQuartile()



class Data():

	def __init__(self, fileName):
		self.fileName = fileName
		self.entries = []
		self.features = []
		try:
			with open(fileName, newline='') as csvFile:
				csvReader = csv.DictReader(csvFile)
				for row in csvReader:
					# for feature in row:
						# if row[feature] == "":
						# 	row[feature] = None
						# if
					self.entries.append(row)
		except Exception as e:
			raise DataError(f"Couldn't retrieve the data from csv file. {str(e)}")
		print(*[row for row in self.entries[:20]],sep="\n\n")



