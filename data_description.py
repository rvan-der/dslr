import csv
import sys


def shortenName(name):
	words = name.split()
	if len(words) == 1:
		return name[:7] + '.' if len(name) > 8 else name
	if len(name) > 8:
		return "".join([word[0] if word.lower() in ["the","of"] else word[0].upper() for word in words])
	return name

		
def quantile(k, q, values, _sort=True):
	if any(map(lambda e:not isinstance(e, float), values)):
		sys.exit("Can't compute the quantile. Some values of the feature are non numeric.")
	if q < 2 or q % 1:
		sys.exit(f"Invalid q value ({q}). Must be int >= 2")
	if k < 1 or k >= q or k % 1:
		sys.exit(f"Invalid k value ({k}) for q={q}. Must be 1 =< int < q")
	if _sort:
		values = sorted(values)
	count = len(values)
	if count < q:
		sys.exit(f"Insufficient values to compute {q}-quantile (count={count})")
	index = count * k / q
	if index % 1:
		return values[int(index)]
	index = int(index)
	return (values[index - 1] + values[index]) / 2


def mean(values):
	if any(map(lambda e:not isinstance(e, float), values)):
		sys.exit("Can't compute the mean. Some values of the feature are non numeric.")
	return sum(self.values) / self.count
	

def lowerQuartile(values, _sort=True):
	return quantile(1, 4, values, _sort=_sort)


def upperQuartile(values, _sort=True):
	return quantile(3, 4, values, _sort=_sort)


def median(values, _sort=True):
	return quantile(1,2, values)