import numpy as np


class LinearRegression:

    def __init__(self, max_iter=200, learning_rate=0.01):
        self.w = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.cost = []

    def fit(self, X, y):
        m, n = X.shape
        n += 1
        X = np.hstack((X, np.ones((m, 1))))
        self.w = np.random.normal(size=(n, 1))
        y = y.reshape((m, 1))
        it = 0
        while it < self.max_iter:
            dw = X.T @ (X @ self.w - y) / m
            self.w = self.w - self.learning_rate * dw
            c = np.power(y - X @ self.w, 2).mean() / 2
            self.cost.append(c)
            it += 1

        return self

    def predict(self, X):
        m, n = X.shape
        n += 1
        X = np.hstack((X, np.ones((m, 1))))
        y_ = X @ self.w
        return y_


class RidgeRegression:

    def __init__(self, max_iter=200, learning_rate=0.001, alpha=1.0):
        self.w = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.cost = []

    def fit(self, X, y):
        m, n = X.shape
        n += 1
        X = np.hstack((X, np.ones((m, 1))))
        self.w = np.random.normal(size=(n, 1))
        y = y.reshape((m, 1))
        it = 0
        while it < self.max_iter:
            dw = (X.T @ (X @ self.w - y) + self.alpha * self.w) / m
            self.w = self.w - self.learning_rate * dw
            c = (np.power(y - X @ self.w, 2) +
                 self.alpha * self.w.T @ self.w).mean() / 2
            self.cost.append(c)
            it += 1

        return self

    def predict(self, X):
        m, n = X.shape
        n += 1
        X = np.hstack((X, np.ones((m, 1))))
        y_ = X @ self.w
        return y_


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


class LogisticRegression:

    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = None
        self.cost = []
        self.acc = []

    def fit(self, X, y):
        m, n = X.shape
        y = y.reshape((m, 1))
        X_ = np.hstack((X, np.ones((m, 1))))
        n += 1
        self.w = np.random.poisson(size=(n, 1))
        it = 0
        while it < self.max_iter:
            dw = X_.T @ (sigmoid(X_ @ self.w) - y) / m
            self.w = self.w - self.learning_rate * dw
            h = sigmoid(X_ @ self.w)
            c = (y * np.log(h) + (1 - y) * np.log(1 - h)).sum() / (-m)
            self.cost.append(c)
            y_ = self.predict(X)
            self.acc.append((y_ == y.squeeze()).mean())
            it += 1
        return self

    def predict(self, X):
        m, n = X.shape
        X_ = np.hstack((X, np.ones((m, 1))))
        n += 1
        y_ = sigmoid(X_ @ self.w)
        y_ = y_.squeeze()
        mask = (y_ >= 0.5)
        y_[mask] = 1
        y_[~mask] = 0
        return y_


class LDA:
    """
    Linear discriminant analysis
    """

    def __init__(self):
        self.u0 = None
        self.u1 = None
        self.sigma0 = None
        self.sigma1 = None
        self.w = None
        self.center0 = None
        self.center1 = None

    def fit(self, X, y):
        mask = (y == 0)
        self.u0 = X[mask].mean(axis=0)
        self.u1 = X[~mask].mean(axis=0)
        self.sigma0 = np.cov(X[mask].T)
        self.sigma1 = np.cov(X[~mask].T)
        u, s, v = np.linalg.svd(self.sigma0 + self.sigma1)
        self.w = v / s @ u.T @ (self.u0 - self.u1)
        self.center0 = self.w @ self.u0
        self.center1 = self.w @ self.u1
        return self

    def predict(self, X):
        """
        w shape = (n,)
        X shape = (m, n)
        """
        p = X @ self.w
        mask = np.absolute(p - self.center0) < np.absolute(p - self.center1)
        p[mask] = 0
        p[~mask] = 1
        return p


class MLDA:

    def __init__(self):
        self.W = None
        self.center = None
        self.u = None
        self.u_mat = None
        self.Sw = None
        self.Sb = None

    def fit(self, X, y):
        m, n = X.shape
        # 求每一类样本的个数
        m_d = dict()
        d = 0
        for elem in y:
            if m_d.get(elem) == None:
                m_d[elem] = 1
                d += 1
            else:
                m_d[elem] += 1

        self.Sw = np.zeros((n, n))
        self.Sb = np.zeros((n, n))
        self.u = X.mean(axis=0)
        u_list = []
        for i in range(d):
            mask = (y == i)
            self.Sw += np.cov(X[mask].T)
            ui = X[mask].mean(axis=0)
            u_list.append(ui)
            self.Sb += m_d[i] * ((ui - self.u) @ (ui - self.u))
        self.u_mat = np.array(u_list)  # shape = d * n
        w, v = np.linalg.eig(np.linalg.inv(self.Sw) @ self.Sb)
        self.W = v[:, :d-1]
        self.center = self.u_mat @ self.W
        return self

    def predict(self, X):
        m = X.shape[0]
        y = np.zeros(m)
        for row, i in zip(X, range(m)):
            projector = row @ self.W
            y[i] = ((projector - self.center) ** 2).sum(axis=1).argmin()
        return y
