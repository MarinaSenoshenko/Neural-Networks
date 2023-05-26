from math import isnan
import logging as log
from operations import printList


def get_data(input_list):
    for i in range(3):
        input_list.pop(0)
    data = [value[2:] for value in input_list]
    return process_data(data)


def has_value(cell):
    return cell not in {'-', 'NaN', '', 'не спускался', '#ЗНАЧ!'}


def process_cell(cell):
    new_cell = cell.replace(',', '')
    if has_value(cell):
        return float(new_cell)
    return float('nan')


def process_data(data):
    prepared_data = []
    for row in data:
        prepared_data.append([process_cell(element) for element in row])
    return prepared_data


def merge_kgf(data, column_names):
    first_kgf_index = len(data[0]) - 2
    second_kgf_index = len(data[0]) - 1
    merged_columns_names = [column_names[first_kgf_index], column_names[second_kgf_index]]
    log.info(f"Target columns {printList(merged_columns_names, ' and ')} were merged")

    for row in data:
        if isnan(row[first_kgf_index]) and not isnan(row[second_kgf_index]):
            row[first_kgf_index] = 1000 * row[second_kgf_index]
    column_names.pop()
    for row in data:
        row.pop()


def is_significant_rows(row, target_variables_count):
    for element in row[-1 * target_variables_count:]:
        if not isnan(element):
            return True
    return False


def filter_rows(data, predicate):
    deleted_rows = []
    i = 0
    while i < len(data):
        if predicate(data[i]):
            deleted_rows.append(data[i])
            del data[i]
        else:
            i += 1
    return deleted_rows


def delete_insignificant_rows(data, target_variables_count, column_names):
    rows_to_delete = filter_rows(data, lambda x: not is_significant_rows(x, target_variables_count))
    log.info(f"{len(rows_to_delete)} count insignificant rows were deleted "
             f"(where values of {printList(column_names[-1 * target_variables_count:], ' and ')} is nan)")
    return rows_to_delete


def numerate_columns(input_list):
    index = 0
    numerated_list = []
    for element in input_list:
        numerated_list.append(f"({str(index)}) {element}")
        index += 1
    return numerated_list
