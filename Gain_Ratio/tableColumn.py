from math import log

from columnOperations import *
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass()
class Column:
    data: list
    column_name: str
    is_categorical: bool
    missing_rate: float
    unique_elements_count: int
    unique_elements_percentage: float
    mode: float
    mean: float
    variance: float
    first_quantile: float
    third_quantile: float

    def __init__(self, data, column_name):
        self.column_name = column_name
        self.data = data
        self.init_column_informaion()

    def init_column_informaion(self):
        self.missing_rate = count_missing_rate(self.data)
        self.unique_elements_count = count_unique_elements(self.data)
        self.mode = count_mode(self.data)
        self.mean = count_arithmetic_middle(self.data)
        self.variance = count_variance(self.data)
        self.first_quantile = count_quantile(self.data, 1)
        self.third_quantile = count_quantile(self.data, 3)
        self.unique_elements_percentage = (self.unique_elements_count / len(self.get_not_nan_elements())) * 100
        self.is_categorical = is_categorical(self.unique_elements_percentage)

    def get_not_nan_elements(self):
        return [element for element in self.data if not isnan(element)]

    def fill_missing_values(self):
        if self.missing_rate < 30:
            filling_value = self.mode if self.is_categorical else self.mean
            for i in range(len(self.data)):
                if isnan(self.data[i]):
                    self.data[i] = filling_value
            self.missing_rate = count_missing_rate(self.data)
            return True
        return False

    def figure_histogram(self):
        sorted_data = sorted(self.data)
        bins_count = int(1 + log(len(sorted_data), 2))
        plt.hist(sorted_data, bins=bins_count, density=False)
        plt.title(self.column_name)
        upper_bound = self.get_upper_bound()
        lower_bound = self.get_lower_bound()
        if upper_bound:
            plt.axvline(x=upper_bound)
        if lower_bound:
            plt.axvline(x=lower_bound)
        plt.show()

    def get_lower_bound(self):
        bound = self.first_quantile - 1.5 * (self.third_quantile - self.first_quantile)
        return bound

    def get_upper_bound(self):
        bound = self.third_quantile + 1.5 * (self.third_quantile - self.first_quantile)
        return bound

    def normalize(self):
        min_value = min(self.data)
        max_value = max(self.data)
        prepared_data = []
        for value in self.data:
            prepared_value = (value - min_value) / (max_value - min_value)
            prepared_data.append(prepared_value)
        self.data = prepared_data
