import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from svm import SVC


def load_data():
    """[load the first two types of iris from the dataset]
    """
    X, y = datasets.load_iris(return_X_y=True)
    idx = y < 2
    y = y[idx]
    y = y.reshape((-1, 1))  # 只有一个维度的可以用-1代替
    X = X[idx]
    return X.T, y.T


def test():
    X, y = load_data()
    X = (X - X.mean(axis=0)) / X.var(axis=0)
    X = np.array([[-1, 0, 1, -2, -1, 0], [-1, 0, 1, 0, 1, 2]])
    y = np.array([[1, 1, 1, 0, 0, 0]])
    plt.scatter(X[0, :3], X[1, :3], c='r')
    plt.scatter(X[0, 3:], X[1, 3:], c='g')
    plt.plot(np.linspace(-3, 2, 50), np.linspace(-3, 2, 50) + 1)
    plt.grid()
    plt.show()
    clf = SVC(C=1.0)
    clf.fit(X, y)
    print(clf.evaluate(X, y))


if __name__ == "__main__":
    test()
