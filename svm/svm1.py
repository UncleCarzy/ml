import numpy as np


class SVC(object):

    def __init__(self, C=1.0, kernel="rbf"):

        assert kernel in (
            "rbf", "linear"), "kernel should be one of ('rbf','linear')\n"
        self.kernel = kernel

        self.C = C
        self.coef = None
        self.dual_coef = None
        self.intercept = None

        self.support_vector = None
        self.n_sv = None
        self.n_feature = None

        self.E = None
        self.K = None
        self.epsilon = 1e-8

    def fit(self, X, y):
        self.n_feature, N = X.shape
        self.K = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, i + 1):
                self.K[i, j] = self.K[j, i] = self.__kernel_function(
                    X[:, i], X[:, j])

        self.__smo(X, y)  # 求得dual_coef

        mask = self.dual_coef > 1e-3
        self.n_sv = mask.sum()
        self.dual_coef = self.dual_coef[mask]
        X_sv, y_sv = X[:, mask], y[mask]
        self.support_vector = (X_sv, y_sv)

        # 只有线性核才可以直接计算出coef
        if "linear" == self.kernel:
            self.coef = X_sv @ (self.dual_coef * y_sv)
        self.intercept = (y_sv - (self.dual_coef * y_sv)
                          @ self.K[mask][:, mask]).mean()
        del self.K
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

    def __smo(self, X, y):

        pass

    def __stopping(self):
        # 判断是否满足KKT条件

        pass

    def __choose(self):
        # 选择违背KKT条件最厉害的变量 alpha1,alpha2

        pass

    def __rbf(self, xi, xj):
        """
        Gaussian kernel function

        Parameters
        ----------
        xi : np.ndarray
            1D-array = vector
        xj : np.ndarray
            1D-array = vector

        Returns
        -------
        float
            product of two vectors tranformed
        """
        variance = 1.0
        diff = xi - xj
        return np.exp(- (diff @ diff) / (2 * variance))

    def __linear(self, xi, xj):
        return xi @ xj

    def __kernel_function(self, xi, xj):
        if "linear" == self.kernel:
            return self.__linear(xi, xj)
        elif "rbf" == self.kernel:
            return self.__rbf(xi, xj)
