import numpy as np
from math import fabs
from random import shuffle


class SVC(object):

    def __init__(self, C=1.0, eps=1e-4, kernel="rbf", max_iter=100):

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
        # 精度范围
        self.eps = eps

        self.max_iter = max_iter
        # 临时变量，训练完会删掉
        self.E = None
        self.K = None
        self.y = None

        self.Done = False

    def fit(self, X, y):
        self.n_feature, N = X.shape
        self.y = y
        self.K = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, i + 1):
                self.K[i, j] = self.K[j, i] = self.__kernel_function(
                    X[:, i], X[:, j])

        self.__smo(X, y)  # 求得dual_coef

        mask = self.dual_coef > self.eps
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
        del self.y
        del self.E
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
        # 序列最小优化算法
        self.__initialize()
        iter = 0
        N = y.shape[0]
        noChangeTimes = 0
        while True:
            paramChanged = 0
            index_list = list(range(N))
            # shuffle(index_list)
            for i in index_list:
                if not self.__check_KKT(i):
                    j = self.__choose_j_by_maxE(i)
                    ai, aj = self.__solve(i, j)
                    if fabs(aj - self.dual_coef[j]) > self.eps:
                        paramChanged += 1
                        self.__update(ai, aj, i, j)
            if paramChanged == 0:
                noChangeTimes += 1
            if noChangeTimes > 10 or iter > self.max_iter:
                break
            print("iter = %d \t paramChanged = %d" % (iter, paramChanged))
            iter += 1
        print("iter = %d" % iter)

    def __initialize(self):
        self.N = self.y.shape[0]
        self.dual_coef = np.zeros(self.N)
        self.intercept = 0.0
        self.E = - self.y.copy()

    def __stopping(self):
        # 判断所有样本点是否满足KKT条件
        for i in range(self.N):
            if not self.__check_KKT(i):
                return False
        return True

    def __choose_j_random(self, i):
        while True:
            j = np.random.choice(self.N, 1)[0]
            if j != i:
                return j

    def __choose_j_by_maxE(self, i):
        max_diff_E = 0.0
        best_j = None
        for j in range(self.N):
            diff = fabs(self.E[i] - self.E[j])
            if j != i and diff > max_diff_E:
                max_diff_E = diff
                best_j = j
        return best_j

    def __check_KKT(self, i):
        # 检查训练样本(xi,yi)是否满足KKT条件
        ai = self.dual_coef[i]
        eps = self.eps
        tmp = self.y[i] * self.__g(i)
        if fabs(ai) < eps and tmp >= 1.0:
            return True
        if 0.0 < ai and ai < self.C and fabs(tmp - 1.0) < eps:
            return True
        if fabs(ai - self.C) < eps and tmp <= 1.0:
            return True
        return False

    def __solve(self, i, j):
        # 解两个变量的子问题
        ai_old = self.dual_coef[i]
        aj_old = self.dual_coef[j]

        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
        # if eta < 1e-8:
        # eta = 1e-4
        diff_E = self.E[i] - self.E[j]
        aj_unc = aj_old + self.y[j] * diff_E / eta

        if self.y[i] != self.y[j]:
            L = max(0, aj_old - ai_old)
            H = min(self.C, self.C + aj_old - ai_old)
        else:
            L = max(0, aj_old + ai_old - self.C)
            H = min(self.C, aj_old + ai_old)

        if aj_unc > H:
            aj = H
        elif L <= aj_unc:
            aj = aj_unc
        else:
            aj = L

        ai = ai_old + self.y[i] * self.y[j] * (aj_old - aj)
        return ai, aj

    def __update(self, ai, aj, i, j, print_flag=False):
        """
        解出子问题之后
            更新 intercept
            更新 dual_coef
            更新 E
        """
        if print_flag:
            print("old: a%d = %f\t a%d = %f" %
                  (i, self.dual_coef[i], j, self.dual_coef[j]))
            print("new: a%d = %f\t a%d = %f\n" % (i, ai, j, aj))

        diff_i = ai - self.dual_coef[i]
        diff_j = aj - self.dual_coef[j]
        intercept_i = - self.E[i] - self.y[i] * self.K[i, i] * \
            diff_i - self.y[j] * self.K[j, i] * diff_j + self.intercept

        intercept_j = - self.E[j] - self.y[i] * self.K[i, j] * \
            diff_i - self.y[j] * self.K[j, j] * diff_j + self.intercept

        if 0.0 < ai and ai < self.C:
            self.intercept = intercept_i
        elif 0.0 < aj and aj < self.C:
            self.intercept = intercept_j
        else:
            self.intercept = (intercept_i + intercept_j) / 2

        self.dual_coef[i] = ai
        self.dual_coef[j] = aj

        for i in range(self.N):
            self.E[i] = self.__g(i) - self.y[i]

    def __g(self, i):
        return (self.dual_coef * self.y) @ self.K[i] + self.intercept

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
