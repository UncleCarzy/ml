import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def load_data():
    X, y = datasets.load_iris(return_X_y=True)
    # 降成2维便于可视化
    X = PCA(2).fit_transform(X)
    mask = y < 2
    X = X[mask]
    y = y[mask]

    y[y == 0] = -1
    X = X.T
    y = y.astype(float)  # 都要是float类型的，整数类型会出现错误
    return X, y


class SVM(object):
    """
    hard margin support vector machine，只能针对线性可分的数据
    binary classifier
    二次规划是用cvxopt求解的
    """

    def __init__(self):
        self.support_vector = []
        self.w = None
        self.b = None

    def fit(self, X, y):
        N = X.shape[1]
        Q = matrix(((y.reshape((-1, 1)) @ y.reshape((1, -1))) * (X.T @ X)))
        p = matrix([-1.0] * N)  # 默认生成列优先的矩阵
        G = matrix(- np.eye(N))
        h = matrix([0.0] * N)
        A = matrix(y.tolist(), (1, N))
        b = matrix(0.0)
        sol = solvers.qp(Q, p, G, h, A, b)

        lam = np.array(sol['x'])
        lam = np.squeeze(lam)
        w = X @ (lam * y)
        mask = np.abs(lam) > 1e-4
        b = (y[mask].reshape((1, -1)) - w.T @ X[:, mask]).mean()

        for index, flag in enumerate(mask):
            if flag:
                self.support_vector.append((X[:, index], y[index]))

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        y = self.w @ X + self.b
        return np.where(y > 0, 1, -1)


if __name__ == "__main__":
    X, y = load_data()
    clf = SVM()
    clf.fit(X, y)
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

    plt.show()
    # plt.savefig("svm\\svm1.png", dpi=200)
