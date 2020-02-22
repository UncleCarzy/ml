import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from svm1 import SVC


def load_data(dataset_name, n_features=2):
    if dataset_name == "iris":
        X, y = datasets.load_iris(return_X_y=True)
    elif dataset_name == "wine":
        X, y = datasets.load_wine(return_X_y=True)
    # 降成2维便于可视化
    X = PCA(n_features).fit_transform(X)
    mask = y < 2
    X = X[mask]
    y = y[mask]
    y[y == 0] = -1.0
    y = y.astype(float)  # 都要是float类型的，整数类型会出现错误
    X, y = shuffle(X, y)
    X = X.T
    return X, y


def test():
    X, y = load_data("iris", 2)
    clf = SVC(kernel="linear")
    clf.fit(X, y)
    y_ = clf.predict(X)
    print(accuracy_score(y_, y))
    print(clf.n_sv)


if __name__ == "__main__":
    sys.stdout = open("out.txt", "w")
    test()
