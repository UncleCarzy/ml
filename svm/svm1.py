import numpy as np


class SVC(object):

    def __init__(self, C=1.0, eps=1e-2, kernel="rbf"):

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
        self.eps = 0.1
        self.tol = 1.0
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
        # 序列最小优化算法
        self.__initialize()
        while True:
            i, j = self.__choose()

            ai, aj = self.__solve(i, j)
            self.__object_value_change(ai, aj, i, j, True)
            self.__update(ai, aj, i, j)

            if self.__stopping():
                break

    def __initialize(self):
        N = self.y.shape[0]
        self.dual_coef = np.zeros(N)
        # self.dual_coef = np.clip(np.random.normal(size=N), 0, self.C)
        self.intercept = 0.0
        self.E = np.zeros(N)
        for k in range(N):
            self.E[k] = self.__g(k) - self.y[k]

    def __stopping(self):
        # 判断是否满足KKT条件
        # print("max: %.3f\t min: %.3f\t mean:%.7f" %
            #   (self.dual_coef.max(), self.dual_coef.min(), self.dual_coef.mean()))
        # 线性约束
        if not abs(self.dual_coef @ self.y) < self.eps:
            return False

        # 边界约束 0.0 <= alpha <= C
        if not (np.all(0.0 <= self.dual_coef) and np.all(self.dual_coef <= self.C)):
            return False
        print("11")
        # 不等式约束
        N = self.y.shape[0]
        for k in range(N):
            if not self.__check_KKT(k):
                return False
        return True

    def __choose(self):
        # 选择违背KKT条件最厉害的变量 alpha1, alpha2
        # 外层循环首先遍历所有 0 < ai < C 的样本点，即在间隔边界上的支持向量点，检验它们是否满足KKT条件
        # 如果这些样本点都满足KKT条件，那么遍历整个训练集，检验它们是否满足KKT条件
        best_i, best_j = None, None
        i = 0
        N = self.y.shape[0]
        while i < N:
            if self.__check_KKT(i):
                j = self.__choose_j_by_maxE(i)
                ai, aj = self.__solve(i, j)
                if self.__object_value_change(ai, aj, i, j):
                    return i, j
                    # best_i, best_j = i, j
                    # break

                a = self.dual_coef
                svp_list = [k for k in range(N) if (k != i and
                                                    0.0 < a[k] and a[k] < self.C)]
                for j in svp_list:
                    ai, aj = self.__solve(i, j)
                    if self.__object_value_change(ai, aj, i, j):
                        return i, j
                        # best_i, best_j = i, j
                        # break

                nsvp_list = [k for k in range(N) if k != i and not(
                    0.0 < a[k] and a[k] < self.C)]
                for j in nsvp_list:
                    ai, aj = self.__solve(i, j)
                    if self.__object_value_change(ai, aj, i, j):
                        return i, j
                        # best_i, best_j = i, j
                        # break
            i += 1
        if best_i == None and best_j == None:
            best_i, best_j = np.random.choice(N, 2)
            # raise Exception("选不出合适的i,j\n")
            # best_j = 1
            # best_i = 0

        return best_i, best_j

    def __choose_j_by_maxE(self, i):
        N = self.y.shape[0]
        max_diff_E = 0.0
        best_j = None
        for j in range(N):
            diff = abs(self.E[i] - self.E[j])
            if j != i and diff > max_diff_E:
                max_diff_E = diff
                best_j = j
        return best_j

    def __object_value_change(self, ai, aj, i, j, flag=False):
        a = np.copy(self.dual_coef)
        y = self.y
        K = self.K
        old_v = a[i] * y[i] * ((a * y) @ K[i]) + \
            a[j] * y[j] * ((a * y) @ K[j]) - (a[i] + a[j])
        a[i], a[j] = ai, aj
        new_v = a[i] * y[i] * ((a * y) @ K[i]) + \
            a[j] * y[j] * ((a * y) @ K[j]) - (a[i] + a[j])
        diff = old_v - new_v

        if flag:
            print("i = %d j = %d diff = %.5f" % (i, j, diff))

        return diff >= self.tol

    def __check_KKT(self, i):
        # 检查训练样本(xi,yi)是否满足KKT条件
        ai = self.dual_coef[i]
        tmp = self.y[i] * self.__g(i)
        if abs(ai) < self.eps and not (tmp >= 1.0):
            return False
        if 0.0 < ai and ai < self.C and not (abs(tmp - 1.0) < self.eps):
            return False
        if abs(ai - self.C) < self.eps and not (tmp <= 1.0):
            return False
        return True

    def __solve(self, i, j):
        # 解两个变量的子问题
        ai_old = self.dual_coef[i]
        aj_old = self.dual_coef[j]

        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
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

    def __update(self, ai, aj, i, j):
        """
        解出子问题之后
            更新 intercept
            更新 dual_coef
            更新 E
        """
        print("old: a%d = %f\t a%d = %f" %
              (i, self.dual_coef[i], j, self.dual_coef[j]))
        print("new: a%d = %f\t a%d = %f\n" % (i, ai, j, aj))
        diff_i = ai - self.dual_coef[i]
        diff_j = aj - self.dual_coef[j]
        intercept_i = - self.E[i] - self.y[i] * self.K[i, i] * \
            diff_i - self.y[j] * self.K[j, i] * diff_j + self.intercept

        intercept_j = - self.E[j] - self.y[i] * self.K[i, j] * \
            diff_i - self.y[j] * self.K[j, j] * diff_j + self.intercept

        self.intercept = (intercept_i + intercept_j) / 2

        self.dual_coef[i] = ai
        self.dual_coef[j] = aj

        self.E[i] = self.__g(i) - self.y[i]
        self.E[j] = self.__g(j) - self.y[j]

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
