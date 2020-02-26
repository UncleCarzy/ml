import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from svm3 import SVC
from svm_by_pkg import SSVM
from svm4 import smoP


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


def test():
    X, y = load_data("iris", 2)
    start = time.time()
    b1, alpha = smoP(X.T, y, 1.0, 1e-10, 500)
    alpha = np.squeeze(np.array(alpha))
    w1 = np.squeeze((alpha * y) @ X.T)
    b1 = np.squeeze(np.array(b1))
    print('time span:', time.time() - start)

    start = time.time()
    clf2 = SSVM()
    clf2.fit_dual_problem(X, y)
    print('time span:', time.time() - start)

    print("coef: clf1 = ", w1, " \t clf2 = ", clf2.coef)
    print("intercept: clf1 = ", b1, " \t clf2 = ", clf2.intercept)

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[0, :], X[1, :], c=y)

    # 画分割线
    def f(x, w, b): return (-w[0] / w[1]) * x - (b / w[1])
    x = np.linspace(-3, 1.5)
    # x = np.linspace(X[0, :].min() / 20, X[0, :].max()/10)
    plt.plot(x, f(x, w1, b1), 'k--', label="SMO")

    w2, b2 = clf2.coef, clf2.intercept
    plt.plot(x, f(x, w2, b2), 'r--', label="QP")
    plt.legend()
    plt.savefig("svm\\svm4.png", dpi=300)


if __name__ == "__main__":
    # sys.stdout = open("svm\\out.txt", "w")
    test()
