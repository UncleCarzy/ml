import numpy as np


class SVC(object):

    def __init__(self, C=1.0, kernel="rbf"):
        self.C = C
        self.kernel = kernel
        self.w = None
        self.b = None
        self.alpha = None
        self.E = None
        self.K = None
        self.epsilon = 1e-8
        self.support_vector = []

    def fit(self, X, y):
        # n_feature, n_sample = X.shape

        pass

    def predict(self, X):
        # 只需计算支持向量所对应的核函数的值，即alpha > 0 的那些项
        return None

    def __smo(self, X, y):
        # 变量初始化
        n_sample = X.shape[1]
        self.alpha = np.zeros(n_sample)
        self.K = np.zeros(n_sample)
        for i, xi in enumerate(i, X):
            for j, xj in enumerate(j, X):
                self.K[i, j] = self.__rbf(xi, xj)

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
