import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def accuracy_metric(actual, predicted):
    """ Accuracy
    :param actual: real target values
    :param predicted: inferred target values
    :return: Accuracy metric
    """

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def mae_metric(actual, predicted):
    """ Mean Absolute Error
    :param actual: real target values
    :param predicted: inferred target values
    :return: MAE metric
    """

    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))


def print_confusion_matrix(unique, matrix):
    print('(A)' + ' '.join(str(x) for x in unique))
    print('(P)---')
    for i, x in enumerate(unique):
        print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))


def confusion_matrix(actual, predicted, print_matrix=True):
    """ Calculate & print confusion matrix
    :param actual: real target values
    :param predicted: inferred target values
    :param print_matrix:
    :return:
    """

    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1

    if print_matrix:
        print_confusion_matrix(unique, matrix)

    return unique, matrix


def rmse_metric(actual, predicted):
    """ Root Mean Squared Error
    :param actual: real target values
    :param predicted: inferred target values
    :return: RMSE metric
    """

    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return np.sqrt(mean_error)


def compute_f1_score(actual, predicted):
    p = compute_precision(actual, predicted)
    r = compute_recall(actual, predicted)
    return 2 * p * r / (p + r)


def compute_precision(actual, predicted):
    matrix = build_confusion_matrix(actual, predicted)
    return np.trace(matrix) / np.sum(matrix, axis=1)


def compute_recall(actual, predicted):
    matrix = build_confusion_matrix(actual, predicted)
    return np.trace(matrix) / np.sum(matrix)


def build_confusion_matrix(actual, predicted, print_matrix=False):
    matrix = pd.crosstab(actual, predicted)
    if print_matrix:
        print(matrix)
    return matrix.to_numpy()


def print_avg_metrics(actual, predicted):
    acc = accuracy_metric(actual, predicted)
    r = compute_recall(actual, predicted)
    p = compute_precision(actual, predicted)
    f1 = compute_f1_score(actual, predicted)
    print("Accuracy:", acc)
    print("Recall:", np.mean(r))
    print("Precision:", np.mean(p))
    print("F1 Score:", np.mean(f1))
