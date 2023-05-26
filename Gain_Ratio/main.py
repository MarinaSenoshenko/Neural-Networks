import logging as log
from matplotlib import pyplot as plt
from operations import printList
from table import Table
import tableOperations as table_operations
from math import fabs
import pandas as pd
from gainRatioAlgotithm import calculate_gain_ratio
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

log.basicConfig(filename='resources/gain_ratio.log', filemode='w', format='%(asctime)s [%(levelname)s] %(message)s',
                level=log.INFO)


def get_correlation_map(table):
    correlation_map = table.get_correlation_table()
    highly_correlated_columns: list[tuple[int, int, int]] = []
    for row_index in range(len(correlation_map) - 2):
        for column_index in range(row_index):
            correlation_rate = correlation_map[row_index][column_index]
            if correlation_rate > 0.85 and not is_components(row_index, column_index, correlation_map):
                highly_correlated_columns.append((row_index, column_index, correlation_rate))
    names = table.get_column_names()

    deleted = []
    for column_tuple in highly_correlated_columns:
        log.info(f"Column {printList([names[column_tuple[0]]], '')} correlated ({round(column_tuple[2], 3)}) with "
                 f"{printList([names[column_tuple[1]]], '')} ")
        deleted.append(delete_one(table, column_tuple[0], column_tuple[1]))

    log.info(f"Columns {printList([str(val) for val in sorted(list(set(deleted)))], ', ')} can be deleted")
    table.delete_correlated_columns(sorted(list(set(deleted))))


def delete_one(table, first_column, second_column):
    gain_ratio = get_gain_ratio(table)
    first_rate = gain_ratio[first_column]
    second_rate = gain_ratio[second_column]
    deleted_column_index = first_column if first_rate < second_rate else second_column
    return deleted_column_index


def is_components(first_column_index, second_column_index, correlation_map):
    for i in range(len(correlation_map) - 2):
        if i == first_column_index or i == second_column_index:
            continue
        if fabs(correlation_map[first_column_index][i] - correlation_map[second_column_index][i]) > 0.3:
            return True
    return False


def process_missing_values(table):
    table.delete_half_empty_columns()
    table.fill_missing_values()


def get_gain_ratio(table):
    ratio = []
    for column in table.columns[:-2]:
        ratio.append(
            calculate_gain_ratio(column.data, table.get_target()))
    return ratio


def get_table_from_input_file():
    input_file = pd.read_excel('resources/ID_data_mass_18122012.xlsx', sheet_name='VU', dtype=str, header=None)
    input_list = input_file.fillna('').values.tolist()
    column_names = table_operations.numerate_columns(input_list[1][2:])
    data = table_operations.get_data(input_list)

    table_operations.merge_kgf(data, column_names)
    target_variables_count = 2
    table_operations.delete_insignificant_rows(data, target_variables_count, column_names)

    table = Table(data, column_names, target_variables_count)
    return table


def figure_characteristic(column_names, values):
    plot_characteristic(column_names, values)


def plot_characteristic(column_names, values):
    fig, axes = plt.subplots()
    axes.barh(column_names, values)
    axes.set_facecolor('floralwhite')
    fig.set_figwidth(13)
    fig.set_figheight(10)
    fig.subplots_adjust(left=0.2)
    plt.show()


def remove(table, dr, column_name, lower_bound, upper_bound):
    dr.extend(table.filter_column(column_name, lambda x: lower_bound < x < upper_bound))


def remove_outliers(table):
    upper_bounds = table.get_upper_bound()
    lower_bounds = table.get_lower_bound()

    table.figure_histograms()
    figure_characteristic(table.column_names, upper_bounds)
    figure_characteristic(table.column_names, lower_bounds)

    column_names = ["(3) Рзаб", "(4) Pлин", "(6) Рзаб", "(7) Рлин", "(8) Туст", "(10) Тзаб",
                    "(12) Дебит газа", "(13) Дебит ст. конд.", "(14) Дебит воды", "(15) Дебит смеси",
                    "(17) Дебит кон нестабильный", "(18) Дебит воды", "(26) Ro_c", "(28) Удельная плотность газа "]
    column_indexes = [3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 23, 24]

    dr = []
    for i in range(len(column_names)):
        remove(table, dr, column_names[i], lower_bounds[column_indexes[i]], upper_bounds[column_indexes[i]])

    log.info(f"{len(dr)} outliers were deleted")
    return dr


def main():
    table = get_table_from_input_file()
    table.find_categorical_columns()
    process_missing_values(table)
    remove_outliers(table)
    get_correlation_map(table)

    table.normalize()
    table.save_in_file()
    table.figure_plot_correlation()


if __name__ == '__main__':
    main()
