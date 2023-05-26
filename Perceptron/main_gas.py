import time
from matplotlib import pyplot as plt
import logging as log
import numpy as np
import csv
import math

from metrics import calc_r_square, calc_roc

log.basicConfig(filename='resources/gas_algorithm.log', filemode='w', format='%(message)s', level=log.INFO)

EPOCH_NUM_WITH_RATE = [
    [1500, 0.005],
    [500, 0.001], [500, 0.0005], [500, 0.0005], [1000, 0.0005], [1000, 0.0005],
    [1000, 0.0005], [1000, 0.0005], [2000, 0.0005], [500, 0.0005], [500, 0.0001], [500, 0.0005],
    [500, 0.0005], [500, 0.0005], [1000, 0.0005], [1000, 0.0005], [1000, 0.0001], [1000, 0.0001],
    [1000, 0.0001], [1000, 0.0001], [1000, 0.0001], [1000, 0.0001], [1000, 0.00005], [1000, 0.00001],
]


def save_weights(file_name: str, layers: tuple[np.ndarray, np.ndarray, np.ndarray]):
    layer_1, layer_2, layer_3 = layers
    np.savez(file_name, l1=layer_1, l2=layer_2, l3=layer_3)


def normalize(array, array_max, array_min, out_max, out_min):
    return (array - array_min) / (array_max - array_min) * (out_max - out_min) + out_min


def zero_nans(array: np.ndarray):
    array[np.isnan(array)] = 0.0


def to_one_row(array: np.ndarray) -> np.ndarray:
    return array.reshape(1, array.shape[0])


def load_data(data_file_name: str = 'resources/data/gas.csv') -> (tuple[np.ndarray, np.ndarray],
                                                                  tuple[np.ndarray, np.ndarray],
                                                                  tuple[np.ndarray, np.ndarray]):
    with open(data_file_name, encoding='UTF-8') as data_file:
        reader = csv.reader(data_file, delimiter=',')
        rows = list(reader)

        data = np.asanyarray(rows, dtype=float)
        dataset: np.ndarray = data[:, :-2]
        targets: np.ndarray = data[:, -2:]

        data_indices = list(range(dataset.shape[0]))

        train_part = 0.7

        train_size = round(len(data_indices) * train_part)
        train_indices = data_indices[:train_size]
        test_indices = data_indices[train_size:]

    train_data = dataset[train_indices].copy()
    test_data = dataset[test_indices].copy()

    kgf_max_train = np.nanmax(targets[train_indices][:, -1:])
    kgf_min_train = np.nanmin(targets[train_indices][:, -1:])

    gtotal_max_train = np.nanmax(targets[train_indices][:, :-1])
    gtotal_min_train = np.nanmin(targets[train_indices][:, :-1])

    train_targets = targets[train_indices].copy()
    train_targets[:, 0] = normalize(train_targets[:, 0], gtotal_max_train, gtotal_min_train, 1, 0)
    train_targets[:, 1] = normalize(train_targets[:, 1], kgf_max_train, kgf_min_train, 1, 0)

    kgf_max_test = np.nanmax(targets[test_indices][:, -1:])
    kgf_min_test = np.nanmin(targets[test_indices][:, -1:])

    gtotal_max_test = np.nanmax(targets[test_indices][:, :-1])
    gtotal_min_test = np.nanmin(targets[test_indices][:, :-1])

    test_targets = targets[test_indices].copy()
    test_targets[:, 0] = normalize(test_targets[:, 0], gtotal_max_test, gtotal_min_test, 1, 0)
    test_targets[:, 1] = normalize(test_targets[:, 1], kgf_max_test, kgf_min_test, 1, 0)

    return (dataset, targets), (train_data, train_targets), (test_data, test_targets)


def make_output_buffer(neurons: tuple[np.ndarray, ...], data: np.ndarray) -> tuple[np.ndarray, ...]:
    return (np.empty(shape=data.shape),) + tuple(
        np.empty(shape=(data.shape[0], neurons[i].shape[1])) for i in range(len(neurons)))


def make_layer(inputs_number: int, outputs_number: int, weight_index: int) -> np.ndarray:
    return np.random.uniform(
        0.01 * math.pow(0.2, weight_index),
        0.02 * math.pow(0.2, weight_index),
        size=(inputs_number, outputs_number))


def make_neurons(*neurons_number) -> (tuple[np.ndarray, ...], tuple[np.ndarray, ...]):
    layers_number: int = len(neurons_number) - 1
    layers_weights: list[np.ndarray] = []
    weight_corrections: list[np.ndarray] = []

    weight_index = layers_number - 1

    for i in range(layers_number):
        inputs_number = neurons_number[i] + 1
        outputs_number = neurons_number[i + 1]

        layers_weights.append(make_layer(inputs_number, outputs_number, weight_index))
        weight_corrections.append(np.empty(shape=(inputs_number, outputs_number)))

        weight_index -= 1

    return tuple(layers_weights), tuple(weight_corrections[::-1])


def weight_correction(current_layer_input: np.ndarray, current_layer_output: np.ndarray,
                      next_layer: np.ndarray, next_layer_delta: np.ndarray) -> (np.ndarray, np.ndarray):
    w_times_delta_sum = next_layer_delta @ next_layer.T
    delta = w_times_delta_sum[:, 1:] * current_layer_output * (1.0 - current_layer_output)

    return delta, current_layer_input.T @ delta


def copy(dst: np.ndarray, src: np.ndarray):
    n = len(dst)
    for i in range(n):
        dst[i] = src[i]


def append_ones_column(array: np.ndarray) -> np.ndarray:
    return np.concatenate(((np.ones(shape=(array.shape[0], 1))), array), axis=1)


def weight_correction_last(output_layer_input: np.ndarray, output_layer_output: np.ndarray,
                           target: np.ndarray) -> (np.ndarray, np.ndarray):
    delta = output_layer_output * (1.0 - output_layer_output) * (output_layer_output - target)
    zero_nans(delta)

    return delta, output_layer_input.T @ delta


def back_propagate(neurons: tuple[np.ndarray, ...], target: np.ndarray,
                   outputs_buffer: tuple[np.ndarray, ...], weight_corrections_buffer: tuple[np.ndarray]):
    layers_number = len(neurons)

    output_layer_input = append_ones_column(outputs_buffer[-2])
    output_layer_output = outputs_buffer[-1]

    delta, correction = weight_correction_last(output_layer_input, output_layer_output, target)

    copy(weight_corrections_buffer[0], correction)
    i: int = 1

    for layer_index in range(2, layers_number + 1):
        cur_layer_input = append_ones_column(outputs_buffer[-layer_index - 1])
        cur_layer_output = outputs_buffer[-layer_index]

        delta, correction = weight_correction(cur_layer_input, cur_layer_output, neurons[-layer_index + 1], delta)

        copy(weight_corrections_buffer[i], correction)
        i += 1


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_layer(layer: np.ndarray, data: np.ndarray) -> np.ndarray:
    return sigmoid(data @ layer)


def feed_forward(layers: tuple[np.ndarray, ...], data: np.ndarray, outputs: tuple[np.ndarray, ...]):
    copy(outputs[0], data)
    for i, layer in enumerate(layers):
        copy(outputs[i + 1], apply_layer(layer, append_ones_column(outputs[i])))


def learn(neurons: tuple[np.ndarray, ...], train_data: np.ndarray, train_target: np.ndarray, learning_rate: float,
          outputs_buffer: tuple[np.ndarray, ...], weight_corr_buffer: tuple[np.ndarray]):
    layers_number = len(neurons)

    for i in range(len(train_data)):
        feed_forward(neurons, to_one_row(train_data[i]), outputs_buffer)
        back_propagate(neurons, to_one_row(train_target[i]), outputs_buffer, weight_corr_buffer)

        idx = len(weight_corr_buffer) - 1
        for j in range(layers_number):
            copy(neurons[j], neurons[j] - weight_corr_buffer[idx] * learning_rate)
            idx -= 1


def calc_result_target(neurons: tuple[np.ndarray, ...], data: np.ndarray, outputs_buffer: tuple[np.ndarray]) -> np.ndarray:
    feed_forward(neurons, data, outputs_buffer)
    return outputs_buffer[-1]


def mse_loss(output: np.ndarray, target: np.ndarray) -> float:
    diff = output - target
    zero_nans(diff)
    return np.mean(diff ** 2)


def train(train_neurons: tuple[np.ndarray, ...], train_dataset: np.ndarray, train_targets: np.ndarray,
          test_dataset: np.ndarray, test_targets: np.ndarray, train_errors: list[float], test_errors: list[float],
          epoch_number: int, learning_rate: float, single_row_outputs_buffer: tuple[np.ndarray, ...],
          train_output_buffer: tuple[np.ndarray, ...], test_output_buffer: tuple[np.ndarray, ...],
          weight_corrections_buffer: tuple[np.ndarray, ...]):
    for i in range(epoch_number):
        learn(train_neurons, train_dataset, train_targets, learning_rate, single_row_outputs_buffer,
              weight_corrections_buffer)

        train_result_target = calc_result_target(train_neurons, train_dataset, train_output_buffer)
        test_result_target = calc_result_target(train_neurons, test_dataset, test_output_buffer)

        train_error = mse_loss(train_result_target, train_targets)
        test_error = mse_loss(test_result_target, test_targets)

        train_errors.append(train_error)
        test_errors.append(test_error)


def root_mse_loss(output: np.ndarray, target: np.ndarray) -> float:
    return mse_loss(output, target) ** 0.5


def plot_results(
        neurons: tuple[np.ndarray, ...],
        all_data: np.ndarray,
        all_target: np.ndarray,
        test_data: np.ndarray,
        test_target: np.ndarray,
        test_errors: list[float],
        train_errors: list[float]):

    plt.plot(train_errors)
    plt.title('Training MS-Error')
    plt.savefig("resources/gases/train_MSE.png")
    plt.show()
    plt.clf()

    plt.plot(test_errors)
    plt.title('Test MS-Error')
    plt.savefig("resources/gases/test_MSE.png")
    plt.show()
    plt.clf()

    output = calc_result_target(neurons, test_data, make_output_buffer(neurons, test_data))
    data_points = list(range(len(test_data)))
    log.info("Test MS-Error: " + f"{mse_loss(output, test_target):.5f}")

    g_total_root_mse_error = root_mse_loss(output[:, 0], test_target[:, 0])
    kgf_root_mse_error = root_mse_loss(output[:, 1], test_target[:, 1])

    log.info("G_total RMS-Error: " + f"{g_total_root_mse_error:.5f}")
    log.info("Kgf RMS-Error: " + f"{kgf_root_mse_error:.5f}")

    g_total_r2_score = calc_r_square(test_target[:, 0], output[:, 0])
    kgf_r2_score = calc_r_square(test_target[:, 1], output[:, 1])

    log.info("G_total R^2 score: " + f"{g_total_r2_score:.5f}")
    log.info("Kgf R^2 score: " + f"{kgf_r2_score:.5f}")

    g_total_fpr, g_total_tpr = calc_roc(test_target[:, 0], output[:, 0])
    kgf_fpr, kgf_tpr = calc_roc(test_target[:, 1], output[:, 1])

    plt.plot(g_total_fpr, g_total_tpr)
    plt.title('G_total ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig("resources/gases/g_total_roc_curve.png")
    plt.show()
    plt.clf()

    plt.plot(kgf_fpr, kgf_tpr)
    plt.title('KGF ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig("resources/gases/kgf_roc_curve.png")
    plt.show()
    plt.clf()

    plt.plot(data_points, output[:, 0], test_target[:, 0])
    plt.title('G_total test approximation')
    plt.savefig("resources/gases/g_total_test_approximation.png")
    plt.show()
    plt.clf()

    plt.plot(data_points, output[:, 1], test_target[:, 1])
    plt.title('KGF test approximation')
    plt.tight_layout()
    plt.savefig("resources/gases/kgf_test_approximation.png")
    plt.show()
    plt.clf()

    output = calc_result_target(neurons, all_data, make_output_buffer(neurons, all_data))
    data_points = list(range(len(all_data)))
    plt.plot(data_points, output[:, 0], all_target[:, 0])
    plt.title('G_total approximation')
    plt.legend(('Predicted', 'Expected'))
    plt.tight_layout()
    plt.savefig("resources/gases/g_total_train_approximation.png")
    plt.show()
    plt.clf()

    plt.plot(data_points, output[:, 1], all_target[:, 1])
    plt.title('KGF approximation')
    plt.legend(('Predicted', 'Expected'))
    plt.tight_layout()
    plt.savefig("resources/gases/kgf_train_approximation.png")
    plt.show()


def main():
    (data, target), (train_data, train_target), (test_data, test_target) = load_data('resources/data/gas.csv')
    zero_nans(data)
    zero_nans(train_data)
    zero_nans(test_data)

    input_layer_number = 15
    first_layer_number = 48
    second_layer_number = 59
    output_layer = 2

    neurons, weight_corrections_buffer = make_neurons(input_layer_number, first_layer_number,
                                                      second_layer_number, output_layer)

    single_row_output_buffer = make_output_buffer(neurons, to_one_row(train_data[0]))
    train_output_buffer = make_output_buffer(neurons, train_data)
    test_output_buffer = make_output_buffer(neurons, test_data)

    train_errors = []
    test_errors = []

    total_epoch_number = 0

    start = time.perf_counter()
    for i, (epoch_number, learning_rate) in enumerate(EPOCH_NUM_WITH_RATE):
        train(neurons, train_data, train_target, test_data, test_target,
              train_errors, test_errors, epoch_number, learning_rate,
              single_row_output_buffer, train_output_buffer, test_output_buffer,
              weight_corrections_buffer)
        total_epoch_number += epoch_number
    log.info("Total number: " + str(total_epoch_number) + " epochs")
    log.info("Time learning: {0} minutes".format((time.perf_counter() - start) / 60))

    plot_results(neurons, data, target, test_data, test_target, train_errors, test_errors)


if __name__ == '__main__':
    main()
