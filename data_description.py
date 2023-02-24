from dslr_exceptions import *


def mean(data):
	return sum(data) / len(data)


def standardDeviation(data):
	print("plop")


def quantile(k, q, data):
	if q < 2 or q % 1:
		raise DataDescriptionError(f"Invalid q value ({q})")
	if k < 1 or k >= q:
		raise DataDescriptionError(f"Invalid k value ({k}) for q={q}")
	count = len(data)
	if count < q:
		raise DataDescriptionError(f"Insufficient data to compute {q}-quantile (count={count})")
	data = sorted(data)
	index = count * k / q
	if index % 1:
		return data[int(index)]
	index = int(index)
	return (data[index - 1] + data[index]) / 2