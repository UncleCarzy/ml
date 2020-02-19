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

    def __init__(self, C=1.0, kernel="linear"):
        assert kernel in ("linear", "rbf"), "kernel should be ('linear','rbf')"
        self.kernel = kernel
        self.support_vector = None
        self.n_sv = None
        self.n_feature = None
        self.dual_coef = None
        self.coef = None
        self.intercept = None
        self.C = C

    def fit_dual_problem(self, X, y):
        self.n_feature, N = X.shape

        K = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, i + 1):
                K[i, j] = K[j, i] = self.__kernel_function(X[:, i], X[:, j])

        Q = matrix((y.reshape((-1, 1)) @ y.reshape((1, -1))) * K)
        p = matrix([-1.0] * N)

        tmp_G = np.vstack((-np.eye(N), np.eye(N)))
        G = matrix(tmp_G)
        tmp_h = ([0.0] * N) + ([self.C] * N)
        h = matrix(tmp_h)

        A = matrix(y.tolist(), (1, N))
        b = matrix(0.0)

        sol = solvers.qp(Q, p, G, h, A, b)
        self.dual_coef = np.squeeze(np.array(sol["x"]))

        mask = self.dual_coef > 1e-3
        self.n_sv = mask.sum()
        self.dual_coef = self.dual_coef[mask]
        X_sv, y_sv = X[:, mask], y[mask]
        self.support_vector = (X_sv, y_sv)

        if "linear" == self.kernel:
            self.coef = X_sv @ (self.dual_coef * y_sv)
        self.intercept = (y_sv - (self.dual_coef * y_sv)
                          @ K[mask][:, mask]).mean()
        return self

    def predict(self, X):
        X_sv, y_sv = self.support_vector
        N = X.shape[1]
        K = np.zeros((self.n_sv, N))
        for i in range(self.n_sv):
            for j in range(N):
                K[i, j] = self.__kernel_function(X_sv[:, i], X[:, j])

        y = (self.dual_coef * y_sv) @ K + self.intercept
        y = np.squeeze(y)
        return np.where(y > 0, 1.0, -1.0)

    def __rbf(self, xi, xj):
        diff = xi - xj
        variance = 1.0
        return np.exp(- (diff @ diff) / (2 * variance))

    def __kernel_function(self, xi, xj):
        if "rbf" == self.kernel:
            return self.__rbf(xi, xj)
        elif "linear" == self.kernel:
            return xi @ xj
