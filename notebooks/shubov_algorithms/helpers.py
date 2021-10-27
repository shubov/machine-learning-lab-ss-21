import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler


def to_binary_features(features, num_bins: int = 3):
    """ Converts features with the same domain to binary
    :param features:
    :param num_bins: number of bins
    :return: np.array of features with feature values converted to binary representation
    """
    assert len(features.shape) == 2

    lower = np.min(features, axis=0)
    upper = np.max(features, axis=0)
    ranges = (upper - lower) / num_bins

    feature_list = []
    for ind_feat in range(features.shape[1]):
        binary_labels = np.digitize(
            features[:, ind_feat],
            [lower[ind_feat] + ranges[ind_feat] * b for b in range(1, num_bins)]
        )
        features_binary = np.zeros((binary_labels.size, binary_labels.max() + 1))
        features_binary[np.arange(binary_labels.size), binary_labels] = 1
        feature_list.append(features_binary)

    return np.hstack(feature_list)


def split_targets_one_to_rest(targets, percentage=None):
    """ Splits the dataset in multiple datasets for each unique class label,
    where that class label is assigned 1 and all others -1.
    Additionally, splits each dataset in test and train datasets.

    :param targets: numpy array of shape (N, 1) with N being the number of samples as in the
        provided features and 1 being the number of target dimensions
    :param percentage: (0..1) the ratio of the length of the test dataset to the length of the entire dataset
    :return: list of tuples: (1) original class label which is transformed to 1,
        (2) np.array of targets if percentage is None, else tuple(np.array of train targets, np.array of test targets)
    """
    unique_classes = np.unique(targets)
    result = []

    for class_name in unique_classes:
        data = [1 if (y == class_name) else -1 for y in targets]
        splitted_data = split_data(data, percentage) if percentage else data
        result.append((class_name, splitted_data))
    return result


def transform_to_binary(values):
    label_names = np.unique(values)
    num_label_names = len(label_names)
    labels2index = {name: ix for name, ix in zip(label_names, range(len(label_names)))}
    features = np.zeros((len(values), num_label_names))
    for index_value in range(len(values)):
        features[index_value,labels2index[values[index_value]]] = 1
    print(features)
    return features


def transform_to_numbers(values):
    transform_targets_to_numbers(values)


def transform_targets_to_numbers(targets):
    """

    :param targets: numpy array of shape (N, 1) with N being the number of samples as in the
        provided features and 1 being the number of target dimensions
    :return: np.array of numerical class labels starting from 0
    """
    label_names = np.unique(targets)
    labels2index = {name: ix for name, ix in zip(label_names, range(len(label_names)))}
    return np.array([labels2index[name] for name in targets])


def split_data(data, percentage=0.7):
    """

    :param data: np.array
    :param percentage: (0..1) the ratio of the length of the train dataset to the length of the entire dataset
    :return: (1) train, (2) test
    """
    data_length = len(data)
    divider = round(percentage * data_length)
    return data[:divider], data[divider:]


def split_data3(data, percentage_train=0.8, percentage_validation=0.1):
    """

    :param data: np.array
    :param percentage_train: (0..1) the ratio of the length of the train dataset to the length of the entire dataset
    :param percentage_validation: (0..1) the ratio of the length of the train dataset to the length of the entire dataset
    :return: (1) train, (2) validation, (3) test
    """
    data_length = len(data)
    divider_train = round(percentage_train * data_length)
    divider_validation = round((percentage_train + percentage_validation) * data_length)
    return data[:divider_train], data[divider_train:divider_validation], data[divider_validation:]


def plot_decision_boundary(X, y, pred_func):
    """ Plots the decision boundary

    :param X: features
    :param y: targets
    :param pred_func: prediction function
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def plot_parallel_coordinates(features, targets, name):
    """

    :param features:
    :param targets:
    :param name: name of the column of the target values (index of the last column)
    """
    data = np.hstack((features, np.expand_dims(targets, -1)))
    parallel_coordinates(pd.DataFrame(data), name)


def scale_train_features(features):
    """ To use for scaling train features (use before scale_test_features function)

    :param features:
    :return: (1) scaler object to pass to the scale_test_features function,
        (2) np.array of scaled train features
    """
    scaler = MinMaxScaler().fit(features)
    return scaler, scaler.transform(features)


def scale_test_features(scaler, features):
    """

    :param scaler: scaler object from scale_train_features function
    :param features:
    :return: np.array of scaled test features
    """
    return scaler.transform(features)


def most_common_class(y):
    """
    :param y: the vector of class labels, i.e. the target
    :returns: (1) the most frequent class label in 'y' and (2) a boolean flag indicating whether y is pure
    """
    y_v, y_c = np.unique(y, return_counts=True)
    label = y_v[np.argmax(y_c)]
    f_is_pure = len(y_v) == 1
    return label, f_is_pure


def pp_float_list(ps):
    """pretty print"""
    return ["%2.3f" % p for p in ps]


def get_k_folds(N, k):
    fold_size = N // k
    idx = np.random.permutation(np.arange(fold_size * k))

    splits = np.split(idx, k)
    folds = []
    for i in range(k):
        te = splits[i]
        tr_si = np.setdiff1d(np.arange(k), i)
        tr = np.concatenate([splits[si] for si in tr_si])
        folds.append((tr.astype(np.int), te.astype(np.int)))
    return folds


def get_bootstrap_folds(N, k, train_fraction=0.8):
    folds = []
    for i in range(k):
        idx = np.random.permutation(np.arange(N))
        # m = int(N * train_fraction)
        tr = np.random.choice(idx, size=N, replace=True)  # When we draw with replacement, the test set will be larger
        te = np.setdiff1d(idx, tr)
        folds.append((tr.astype(np.int), te.astype(np.int)))
    return folds
