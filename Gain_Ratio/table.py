from operations import transpose
from tableColumn import Column
import pandas as pd
import matplotlib.pyplot as plt
import logging
from math import isnan
from operations import printList


class Table:
    columns = []
    target_variables_count: int
    column_names: list[str]

    def __init__(self, data, column_names, target_variables_count):
        self.column_names = column_names
        self.target_variables_count = target_variables_count
        transposed_data = transpose(data)
        self.init_columns(transposed_data, column_names)

    def init_columns(self, transposed_data, column_names):
        self.columns.clear()
        for i in range(len(transposed_data)):
            current_column = Column(transposed_data[i], column_names[i])
            self.columns.append(current_column)

    def filter_column(self, column_name, predicate):
        column_index = self.column_names.index(column_name)
        data = self.get_data()
        rows = transpose(data)

        deleted_rows = []
        index = 0
        while index < len(rows):
            row = rows[index]
            if isnan(row[-2]) and not predicate(row[column_index]):
                deleted_rows.append(row)
                rows.pop(index)
                continue
            index += 1
        self.init_columns(transpose(rows), self.column_names)
        return deleted_rows

    def get_column_index(self, column_name):
        for i in range(len(self.columns)):
            column = self.columns[i]
            if column_name == column.get_name():
                return i

    def delete_correlated_columns(self, deleted):
        for i in range(len(deleted)):
            self.delete_colums_from_name(deleted[i])

    def delete_colums_from_name(self, index):
        for i in range(len(self.columns)):
            if self.column_names[i].count(str(index)):
                deleted_column = self.columns.pop(i)
                self.column_names.pop(i)
                return deleted_column

    def delete_column(self, index):
        deleted_column = self.columns.pop(index)
        self.column_names.pop(index)
        return deleted_column

    def get_dictionary(self):
        names = self.get_column_names()
        data = self.get_data()
        dictionary = {names[i]: data[i] for i in range(len(data))}
        return dictionary

    def filter_columns(self, predicate):
        deleted_columns = []
        index = 0
        while index < self.get_off_target_variables_count():
            column = self.columns[index]
            if not predicate(column):
                deleted_column = self.delete_column(index)
                deleted_columns.append(deleted_column)
                continue
            index += 1
        return deleted_columns

    def get_off_target_variables(self):
        return self.columns[:-2]

    def figure_plot_correlation(self):
        df = pd.DataFrame(self.get_dictionary())
        f = plt.figure(figsize=(15, 12))
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=10, rotation=0)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.show()

    def get_correlation_table(self):
        df = pd.DataFrame(self.get_dictionary())
        d = df.corr().to_dict()
        matrix = []
        names = self.column_names
        for name1 in names:
            row = []
            for name2 in names:
                row.append(d[name1][name2])
            matrix.append(row)
        return matrix

    def get_column_names(self):
        return [column.column_name for column in self.columns]

    def get_data(self):
        return [column.data for column in self.columns]

    def get_off_target_variables_count(self):
        return len(self.columns) - self.target_variables_count

    def fill_missing_values(self):
        filled_columns = []
        for column in self.get_off_target_variables():
            was_filled = column.fill_missing_values()
            if was_filled:
                filled_columns.append(column)
        filled_columns_names = [column.column_name for column in filled_columns]
        logging.info(f"Missing values of columns {printList(filled_columns_names, ', ')} were filled")

    def delete_half_empty_columns(self):
        max_missing_rate_count = 30
        deleted_columns = self.filter_columns(lambda column: column.missing_rate <= max_missing_rate_count)
        deleted_columns_names = [column.column_name for column in deleted_columns]
        logging.warning(f"Columns {printList(deleted_columns_names, ', ')} "
                        f"(with missing rate > {max_missing_rate_count})  were deleted")

    def find_categorical_columns(self):
        categorical_columns = [column.column_name for column in self.columns if column.is_categorical]
        logging.warning(
            f"Find {len(categorical_columns)} categorical columns (< 24% unique values): "
            f"{printList(categorical_columns, ', ')}")
        return categorical_columns

    def get_target(self):
        values = []
        table_data = transpose(self.get_data()[-2:])
        for row in table_data:
            values.append((row[0], row[1]))
        return values

    def get_upper_bound(self):
        return [column.get_upper_bound() for column in self.columns]

    def get_lower_bound(self):
        return [column.get_lower_bound() for column in self.columns]

    def figure_histograms(self):
        for column in self.columns:
            column.figure_histogram()

    def save_in_file(self):
        df = pd.DataFrame(self.get_dictionary())
        df.to_csv('resources/result.csv', index=False)

    def normalize(self):
        for column in self.columns:
            column.normalize()
