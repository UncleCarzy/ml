import numpy as np


def count_values(x):
    """
    Parameters
    ----------
    x : numpy.ndarray
        1D-array

    Returns
    -------
    dict
        a -> b
        b is the times of a accuring in  x
    """
    assert isinstance(x, np.ndarray), "x must be a 1D array."
    table = dict()
    for v in x:
        if table.get(v) == None:
            table[v] = 1
        else:
            table[v] += 1
    return table


def are_samples_all_same(X):
    flag = True
    first_row = X[0:, ]
    for row in X:
        if not (first_row == row).all():
            flag = False
            break
    return flag


def majority(table):
    majority_class = None
    majority_num = 0
    for key, value in table.items():
        if value > majority_num:
            majority_num = value
            majority_class = key
    return majority_class
