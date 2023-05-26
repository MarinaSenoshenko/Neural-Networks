import numpy as np


def calc_r_square(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    sse = np.sum((y_true - y_pred) ** 2)
    ssr = np.sum((y_pred - np.mean(y_true)) ** 2)
    return 1 - (sse / ssr)


def calc_f_measure(y_true, y_pred, threshold=0.5):
    y_pred = np.where(y_pred > threshold, 1, 0)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calc_roc(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    thresholds = np.sort(np.unique(y_pred))[::-1]
    fpr = np.zeros(len(thresholds) + 2)
    tpr = np.zeros(len(thresholds) + 2)
    fpr[-1] = 1
    tpr[-1] = 1
    for i, threshold in enumerate(thresholds):
        y_pred_binary = np.where(y_pred >= threshold, 1, 0)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        if (tp + fp) == 0 or (tp + fn) == 0:
            fpr[i] = 0
            tpr[i] = 0
        else:
            fpr[i] = fp / (fp + tn)
            tpr[i] = tp / (tp + fn)
    return fpr, tpr
