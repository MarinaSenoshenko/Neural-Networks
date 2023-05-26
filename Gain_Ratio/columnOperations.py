import statistics
from math import isnan, sqrt
import numpy as np


def count_missing_rate(column):
    elements = [element for element in column if not isnan(element)]
    return round((1 - (len(elements) / len(column))) * 100, 3)


def count_unique_elements(column):
    unique_elements = set([element for element in column if not isnan(element)])
    return len(unique_elements)


def count_mode(column):
    elements = [element for element in column if not isnan(element)]
    return statistics.multimode(elements)[0]


def count_arithmetic_middle(column):
    elements = [element for element in column if not isnan(element)]
    return np.mean(elements)


def count_variance(column):
    elements = [element for element in column if not isnan(element)]
    return sqrt(np.var(elements))


def count_quantile(column, number):
    elements = [element for element in column if not isnan(element)]
    return np.percentile(elements, number * 25)


def is_categorical(unique_values):
    return unique_values < 24
