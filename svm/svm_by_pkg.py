import numpy as np
from cvxopt import matrix, solvers


class HSVM(object):
    """
    hard margin support vector machine，只能针对线性可分的数据
    binary classifier
    二次规划是用cvxopt求解的
    """

    def __init__(self):
        self.support_vector = []
        self.w = None
        self.b = None

    def fit_dual_problem(self, X, y):
        # 解的是对偶问题，不是原问题
        N = X.shape[1]
        Q = matrix(((y.reshape((-1, 1)) @ y.reshape((1, -1))) * (X.T @ X)))
        p = matrix([-1.0] * N)  # 默认生成列优先的矩阵
        G = matrix(- np.eye(N))
        h = matrix([0.0] * N)
        A = matrix(y.tolist(), (1, N))
        b = matrix(0.0)
        try:
            sol = solvers.qp(Q, p, G, h, A, b)
        except:
            raise Exception("只能处理线性可分的数据集\n")
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

    def fit_primal_problem(self, X, y):
        n, N = X.shape
        # W = [w;b]
        tmp = np.eye(n + 1)
        tmp[n, n] = 0.0
        Q = matrix(tmp)
        p = matrix([0.0] * (n + 1))
        tmp = np.vstack((X, np.ones((1, N))))
        G = matrix((- y * tmp).T)
        h = matrix([-1.0] * N)

        try:
            sol = solvers.qp(Q, p, G, h)
        except:
            raise Exception("只能处理线性可分的数据集\n")
        W = np.squeeze(np.array(sol['x']))
        self.w = W[:n]
        self.b = W[n]
        mask = np.abs(1 - y * (self.w @ X + self.b)) <= 1e-7
        for index, flag in enumerate(mask):
            if flag:
                self.support_vector.append((X[:, index], y[index]))
        return self

    def predict(self, X):
        y = self.w @ X + self.b
        return np.where(y > 0, 1, -1)


class SSVM(object):
    """
    soft margin support vector machine
    binary classifier
    """

    def __init__(self, C=1.0):
        self.support_vector = []
        self.w = None
        self.b = None
        self.C = C

    def fit_dual_problem(self, X, y):
        N = X.shape[1]

        Q = matrix((y.reshape((-1, 1)) @ y.reshape((1, -1))) * (X.T @ X))
        p = matrix([-1.0] * N)

        tmp_G = np.vstack((-np.eye(N), np.eye(N)))
        G = matrix(tmp_G)
        tmp_h = ([0.0] * N) + ([self.C] * N)
        h = matrix(tmp_h)

        A = matrix(y.tolist(), (1, N))
        b = matrix(0.0)

        sol = solvers.qp(Q, p, G, h, A, b)
        lam = np.array(sol["x"])
        lam = np.squeeze(lam)

        self.w = X @ (lam * y)

        mask = lam > 0.0
        self.b = (y[mask] - self.w @ X[:, mask]).mean()
        for index, flag in enumerate(mask):
            if flag:
                self.support_vector.append((X[:, index], y[index]))

        return self

    def predict(self, X):
        y = self.w @ X + self.b
        return np.where(y > 0, 1.0, -1.0)
