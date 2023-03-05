'''Statistics functions'''

import numbers
import numpy as np
import math


class TinyStatistician:

    def __init__(self) -> None:
        pass

    def mean(self, x):
        '''Computes the mean value of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'mean' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        total = 0.0
        for item in x:
            total += item
        return float(total / len(x))

    def median(self, x):
        '''Computes the median value of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print(
                "Function 'median' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        if len(y) % 2 == 1:
            return float(y[len(y) // 2])
        return float((y[len(y) // 2] + y[-1 + len(y) // 2]) / 2)

    def quartile(self, x):
        '''Computes the first and third quartiles of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print(
                "Function 'quartile' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        lim1 = min(0, int(math.ceil(0.25 * len(y))) - 1)
        lim2 = min(0, int(math.ceil(0.75 * len(y))) - 1)
        return [y[lim1], y[lim2]]

    def percentile(self, x, p):
        '''Computes the expected percentile of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print(
                "Function 'percentile' takes only a list or an array of numbers as parameter")
            return None
        if p not in range(0, 101):
            print("The requested percentile must be between 0 and 100")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        if p == 0:
            return float(y[0])
        res = max(0, int(math.ceil(p * (len(y)) / 100)) - 1)
        return float(y[int(res)])

    def var(self, x):
        '''Computes the variance of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'var' takes only a list or an array of numbers as parameter")
            return None
        if len(x) <= 1:
            return None
        mean = self.mean(x)
        return sum((item - mean) ** 2 for item in x) / (len(x) - 1)
        

    def std(self, x):
        '''Computes the standard deviation of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'std' takes only a list or an array of numbers as parameter")
            return None
        var = self.var(x)
        if not var:
            return None
        return var ** 0.5

    def min(self, x):
        '''Return the min of the list'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'min' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        res = x[0]
        for item in x:
            if item < res:
                res = item
        return res

    def max(self, x):
        '''Return the max of the list'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'max' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        res = x[0]
        for item in x:
            if item > res:
                res = item
        return res