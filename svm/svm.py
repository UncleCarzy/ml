import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def rbf(gamma, x, xc):
    diff = x - xc
    return np.exp(- (diff * diff).sum(axis=0) / (2 * gamma * gamma)).reshape((1, -1))


class SVC:
    """[C-SVM: support vector machine for classification]
    """

    def __init__(self, C=1.0, kernel="rbf", gammaType="auto"):
        """[summary]

        Keyword Arguments:
            C {float} -- [regulariztion parameter,the strength of the regulariztion is inversely proportional to C] (default: {1.0})
            kernel {str} -- [specifies the kernel type to be used in the algorithm."rbf": Radial Basis Function,高斯核函数] (default: {"rbf"})
            gamma {str} -- [the parameter for kernel "rbf"] (default: {"auto"})
        """
        # the parameters for primal problem
        self.C = C
        self.weight = None
        self.b = None
        self.xi = None  # 松弛变量
        self.X = None  # 输入
        self.y = None  # 输入

        # the parameters for dual problem
        self.alpha = None  # 拉格朗日乘子
        self.Q = None  # Qij = yi * yj * K(xi,xj)
        self.K = None  # 经核函数映射后的高维“特征矩阵” kernel matrix(PSD)
        self.gredient = None  # 梯度
        self.p = None
        self.tao = 1e-8
        self.epslion = 1e-4  # stopping criteria

        self.m = None  # 训练集数据个数
        self.nFeatures = None
        self.gammaType = gammaType
        self.gamma = 1.0

    def __prepare(self, X, y):
        # 私有方法
        self.nFeatures, self.m = X.shape
        self.X = X
        self.y = y
        self.y[y == 0] = -1
       # print(self.y)
        # 设置weight的shape
        self.weight = np.zeros((self.m, 1))

        # 设置rbf中的gamma
        if self.gammaType == "auto":
            self.gamma = 1 / self.nFeatures

        # 计算kernel matrix
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[i, :] = rbf(self.gamma, X, X[:, i, None])
       # print("K shape: ", self.K.shape)

        # 计算Q
        self.Q = y.T @ y * self.K
        #print("Q shape: ", self.Q.shape)

        # 初始化alpha
        self.p = - np.ones((self.m, 1))
        self.alpha = np.zeros((self.m, 1))
        self.gredient = np.zeros((self.m, 1)) + self.p

    def __stoppingCondition(self):
        epslion = self.epslion
        self.Iup = []
        self.Ilow = []
        for t in range(self.m):
            if(self.alpha[t, 0] < self.C and self.y[0, t] == 1) or (self.alpha[t, 0] > 0 and self.y[0, t] == -1):
                self.Iup.append(t)
            if(self.alpha[t, 0] < self.C and self.y[0, t] == -1) or (self.alpha[t, 0] > 0 and self.y[0, t] == 1):
                self.Ilow.append(t)

        idx = self.Iup[0]
        mmax = - self.y[0, idx] * self.gredient[idx, 0]
        for t in self.Iup:
            tmp = - self.y[0, t] * self.gredient[t, 0]
      #      print(tmp)
            if tmp > mmax:
                mmax = tmp

        idx = self.Ilow[0]
        mmin = - self.y[0, idx] * self.gredient[idx, 0]
        for t in self.Ilow:
            tmp = - self.y[0, t] * self.gredient[t, 0]
            if tmp < mmin:
                mmin = tmp
      #  print("mmax = %f mmin = %f" % (mmax, mmin))

        if mmax - mmin < epslion:
            return True
        else:
            return False

    def __workingSetSelect(self):
        """[working set select]
        """

        idx = 0
        maxi = -10000.0
        for t in self.Iup:
            tmp = - self.y[0, t] * self.gredient[t, 0]
            if tmp >= maxi:
                maxi = tmp
                idx = t
        i = idx

        idx = 0
        mini = 1.0
        for t in self.Ilow:
            tmpg = - self.y[0, t] * self.gredient[t, 0]
            if tmpg < maxi:
                ats = self.K[i, i] + self.K[t, t] - 2 * self.K[i, t]
                bts = maxi - tmpg
                if ats > 0:
                    ats_ = ats
                else:
                    ats_ = self.tao
                tmp = - (bts ** 2) / ats_  # 是个负数
                if tmp < mini:
                    idx = t
                    mini = tmp
        j = idx
        return i, j

    def __twoVariableSubProblem(self, i, j):
        aij = self.Q[i, i] + self.Q[j, j] - 2 * self.Q[i, j]
        if aij <= 0:
            aij = self.tao
        if self.y[0, i] != self.y[0, j]:
            delta = (- self.gredient[i] - self.gredient[j]) / aij
            diff = self.alpha[i] - self.alpha[j]
            self.alpha[i] += delta
            self.alpha[j] += delta
            if diff > 0:
                if self.alpha[j] < 0:  # region 3
                    self.alpha[j] = 0
                    self.alpha[i] = diff
                elif self.alpha[i] > self.C:  # region 1
                    self.alpha[i] = self.C
                    self.alpha[j] = self.C - diff
            else:
                if self.alpha[i] < 0:  # region 4
                    self.alpha[i] = 0
                    self.alpha[j] = - diff
                elif self.alpha[j] > self.C:  # region 2
                    self.alpha[j] = self.C
                    self.alpha[i] = self.C + diff
        else:  # self.y[i] = self.y[j]
            delta = (-self.gredient[i] + self.gredient[j]) / aij
            sum = self.alpha[i] + self.alpha[j]
            self.alpha[i] += delta
            self.alpha[j] -= delta
            if sum > self.C:
                if self.alpha[i] > self.C:  # region 1
                    self.alpha[i] = self.C
                    self.alpha[j] = - self.C + sum
                elif self.alpha[j] > self.C:  # region 2
                    self.alpha[j] = self.C
                    self.alpha[i] = - self.C + sum
            else:  # sum <= self.C
                if self.alpha[j] < 0:  # region 3
                    self.alpha[j] = 0
                    self.alpha[i] = sum
                elif self.alpha[i] < 0:  # region 4
                    self.alpha[i] = 0
                    self.alpha[j] = sum

    def __smo(self):
        """
        1.初始化alpha为零向量，计算初始梯度
        2.如果alpha_k是stationary point，就跳出循环，结束求解。
        否则，wws选择i，j，解两个变量的子问题
        3.aij > 0 ，则求解子问题一
         否则，求解子问题二
        4.更新参数alpha 和 gredient，跳到2
        """
        self.alpha = np.zeros((self.m, 1))
        self.gredient = np.zeros((self.m, 1)) + self.p
        print(self.Q)
        print(self.K)
        while not self.__stoppingCondition():
            i, j = self.__workingSetSelect()
            aij = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
            print(aij)
            print("i = %d j = %d" % (i, j))
            self.__twoVariableSubProblem(i, j)
            self.gredient[i] = self.Q[i, ] @ self.alpha + self.p[i]
            self.gredient[j] = self.Q[j, ] @ self.alpha + self.p[j]

    def fit(self, X, y):
        self.__prepare(X, y)
        self.__smo()
        # weight算不出来的，只能算b（利用支持向量计算）
        self.svidx = []
        for i in range(self.m):
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                self.svidx.append(i)
        self.b = (self.y[0, self.svidx] - (self.alpha[self.svidx, 0] @ self.y[0,
                                                                              self.svidx] * self.Q[self.svidx, self.svidx]).sum(axis=0)).mean()
        print(self.alpha)
        print("b = %f" % self.b)

    def predict(self, X):
        m = X.shape[1]
        nsv = len(self.svidx)
        tmp_K = np.zeros((nsv, m))
        for i in range(nsv):
            tmp_K[i, :] = rbf(self.gamma, X, self.X[:, i, None])
        y_ = (self.alpha[self.svidx, 0] @ self.y[0, self.svidx]
              * tmp_K).sum(axis=0, keepdims=True) + self.b
        y_[y_ < 0] = -1
        y_[y_ >= 0] = 1
        return y_

    def evaluate(self, X, y):
        y_ = self.predict(X)
        print(y)
        print(y_)
        return (y_ == y).mean()
