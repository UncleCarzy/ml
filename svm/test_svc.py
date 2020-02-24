import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from svm1 import SVC
from svm_by_pkg import SSVM


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
    # X, y = shuffle(X, y)
    X = X.T
    return X, y


def plot_split_line(clf, X, y, filename):
    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[0, :], X[1, :], c=y)

    # 画分割线
    x = np.linspace(-4, 2)
    w, b = clf.coef, clf.intercept
    def f(x): return (-w[0] / w[1]) * x - (b / w[1])
    plt.plot(x, f(x), 'k--')
    plt.savefig(filename)


def test():
    X, y = load_data("iris", 2)
    clf = SVC(C=1.0, kernel="linear", eps=1e-4, max_iter=1000)
    clf.fit(X, y)
    y_ = clf.predict(X)
    print(accuracy_score(y_, y))
    print(clf.n_sv)
    print(clf.dual_coef)
    print(clf.coef)
    print(clf.intercept)
    plot_split_line(clf, X, y, "svm\\1.png")

    clf2 = SSVM()
    clf2.fit_dual_problem(X, y)
    y_ = clf2.predict(X)
    print(accuracy_score(y_, y))
    plot_split_line(clf2, X, y, "svm\\2.png")


if __name__ == "__main__":
    # sys.stdout = open("svm\\out.txt", "w")
    test()
