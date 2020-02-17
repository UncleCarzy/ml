import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from svm_by_pkg import HSVM, SSVM


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
    X = X.T
    y = y.astype(float)  # 都要是float类型的，整数类型会出现错误
    return X, y


def test1():
    # 硬间隔
    X, y = load_data("iris", 2)
    clf = HSVM()
    # clf.fit_dual_problem(X, y)
    clf.fit_primal_problem(X, y)
    y_ = clf.predict(X)
    print(accuracy_score(y_, y))
    print(clf.support_vector)

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[0, :], X[1, :], c=y)

    # 画分割线
    x = np.linspace(-3, 0.5)
    w, b = clf.w, clf.b
    def f(x): return (-w[0] / w[1]) * x - (b / w[1])
    plt.plot(x, f(x), 'k--')

    # plt.show()
    plt.savefig("svm\\hard_svm_by_pkg.png", dpi=200)


def test2():
    # 软间隔
    X, y = load_data("wine", 2)
    clf = SSVM()
    clf.fit_dual_problem(X, y)
    y_ = clf.predict(X)
    print(accuracy_score(y_, y))
    print(len(clf.support_vector))

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[0, :], X[1, :], c=y)

    # 画分割线
    x = np.linspace(10.0, X[0, :].max() / 10)
    w, b = clf.w, clf.b
    def f(x): return (-w[0] / w[1]) * x - (b / w[1])
    plt.plot(x, f(x), 'k--')
    # plt.show()
    plt.savefig("svm\\soft_svm_by_pkg.png", dpi=200)


if __name__ == "__main__":
    test1()
    # test2()
