from math import log, isnan


def get_dictionary(data):
    dictionary = {}
    unique_values = set([value for value in data if not isnan(value)])
    size_unique_values = len(unique_values)
    k = 1 + int(log(size_unique_values, 2))
    max_value = max(unique_values)
    min_value = min(unique_values)
    step = (max_value - min_value) / k
    previous_value = min_value
    for i in range(k):
        dictionary[i] = (previous_value, previous_value + step)
        if i == 0:
            dictionary[i] = (previous_value - 1, previous_value + step)
        if i == k - 1:
            dictionary[i] = (previous_value, previous_value + step + 1)
        previous_value += step
    return dictionary


def get_target_dictionary(data):
    dictionary = {}
    second_column_dictionary = get_dictionary([value[1] for value in data if not isnan(value[1])])
    first_column_unique_values = set([value[0] for value in data if not isnan(value[0])])
    count = 0
    for val in first_column_unique_values:
        for i in range(len(second_column_dictionary.values())):
            dictionary[count] = (val, second_column_dictionary[i])
            count += 1
    return dictionary


def calculate_frequensy(data, bounds):
    frequensy = 0
    for value in data:
        if is_belong_to_dictiionary(value, bounds):
            frequensy += 1
    return frequensy


def part_of_information(frequensy, data_capacity):
    prop = frequensy / data_capacity
    return -1 * prop * log(prop, 2)


def calculate_column_info(column):
    column_elements = get_dictionary(column)
    info = 0
    for i in range(len(column_elements.values())):
        target_bounds = column_elements[i]
        total_count = 0
        current_frequensy = 0
        for row in column:
            total_count += 1
            if is_belong_to_dictiionary(row, target_bounds):
                current_frequensy += 1
        if current_frequensy != 0:
            info += part_of_information(current_frequensy, total_count)
    return info


def calculate_info(input_column, input_bounds, target_columns):
    target_elements = get_target_dictionary(target_columns)
    info = 0
    for i in range(len(target_elements.values())):
        target_bounds = target_elements[i]
        current_frequensy = 0
        capacity = 0
        for j in range(len(input_column)):
            if is_belong_to_dictiionary(input_column[j], input_bounds):
                capacity += 1
                if is_belong_to_target_dictionary(target_columns[j], target_bounds):
                    current_frequensy += 1
        if current_frequensy != 0:
            info += part_of_information(current_frequensy, capacity)
    return info


def is_belong_to_target_dictionary(value, input_bounds):
    is_belong_to_first_column = (value[0] == input_bounds[0])
    is_belong_to_second_column = (input_bounds[1][0] < value[1] < input_bounds[1][1])
    return is_belong_to_first_column and is_belong_to_second_column


def is_belong_to_dictiionary(value, input_bounds):
    return input_bounds[0] <= value < input_bounds[1]


def calculate_info_x(input_column, target_columns):
    input_column_elements = get_dictionary(input_column)
    result_sum = 0
    for i in range(len(input_column_elements.values())):
        result_range = input_column_elements[i]
        frequensy = calculate_frequensy(input_column, result_range)
        result_sum += (frequensy / len(input_column)) * calculate_info(input_column, result_range, target_columns)
    return result_sum


def calculate_target_info(target_columns):
    target_elements = get_target_dictionary(target_columns)
    info = 0
    for i in range(len(target_elements.values())):
        total_count = 0
        current_frequency = 0
        for row in target_columns:
            total_count += 1
            if is_belong_to_target_dictionary(row, target_elements[i]):
                current_frequency += 1
        if current_frequency != 0:
            info += part_of_information(current_frequency, total_count)
    return info


def calculate_gain_ratio(input_column, target_columns):
    info = calculate_target_info(target_columns)
    info_x = calculate_info_x(input_column, target_columns)
    split_info = calculate_column_info(input_column)
    return (info - info_x) / split_info
