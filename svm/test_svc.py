import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from svm import SVC
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


def plot_split_line(dataName, clf1, clf2, X, y, filename):
    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[0, :], X[1, :], c=y)

    # 画分割线
    def f(x, w, b): return (-w[0] / w[1]) * x - (b / w[1])

    if dataName == "iris":
        x = np.linspace(-3, 1.5)
    elif dataName == "wine":
        x = np.linspace(0, 100)

    w1, b1 = clf1.coef, clf1.intercept
    plt.plot(x, f(x, w1, b1), 'k--', label="SMO")

    w2, b2 = clf2.coef, clf2.intercept
    plt.plot(x, f(x, w2, b2), 'r--', label="QP")
    plt.legend()
    plt.savefig(filename, dpi=300)


def test(dataName):
    X, y = load_data(dataName, 2)
    start = time.time()
    clf1 = SVC(C=1.0, kernel="linear", eps=1e-10, max_iter=2000)
    clf1.fit(X, y)
    print('time span:', time.time() - start)
    y_1 = clf1.predict(X)

    start = time.time()
    clf2 = SSVM()
    clf2.fit_dual_problem(X, y)
    print('time span:', time.time() - start)
    y_2 = clf2.predict(X)

    print("ACC: clf1 = %.4f \t clf2 = %.4f" %
          (accuracy_score(y_1, y), accuracy_score(y_2, y)))
    print("n_sv: clf1 = %d \t clf2 = %d" % (clf1.n_sv, clf2.n_sv))
    print("dual_coef:")
    print("clf1 = ", clf1.dual_coef)
    print("clf2 = ", clf2.dual_coef)
    print("coef: clf1 = ", clf1.coef, " \t clf2 = ", clf2.coef)
    print("intercept: clf1 = ", clf1.intercept, " \t clf2 = ", clf2.intercept)

    plot_split_line(dataName, clf1, clf2, X, y,
                    "svm\\" + dataName + "_SVC.png")


if __name__ == "__main__":
    # sys.stdout = open("svm\\out.txt", "w")
    test("iris")
    # test("wine")
