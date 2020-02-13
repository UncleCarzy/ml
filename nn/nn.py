import numpy as np
from sklearn.utils.random import sample_without_replacement
from math import exp


class NerualNetwork:

    def __init__(self, layer_list=[], solver="gd", learning_rate_init=0.01, lr_decay=None, beta1=0.9, beta2=0.9, C=1.0, max_iter=200):

        assert solver in (
            "gd", "mgd", "RMSprop"), "solve should be one of ('gd','mgd','RMSprop')\n"

        # SLR: StepLR
        # ELR:ExponentialLR
        # MLR:MultiStepLR
        # CLR:CosineAnnealingLR
        assert lr_decay in (None, "SLR", "ELR", "MSLR", "CALR")

        self.solver = solver
        self.beta1 = beta1
        self.beta2 = beta2
        self.epslion = 1e-8

        self.lr_decay = lr_decay
        self.learning_rate_init = learning_rate_init
        self.learning_rate = self.learning_rate_init
        self.C = C

        self.max_iter = max_iter

        self.layer_list = layer_list
        self.n_layers = len(layer_list)
        self.n_outputs = layer_list[-1]
        self.loss = []

        self.weights = {}
        self.bias = {}
        self.Vw = {}
        self.Vb = {}
        self.tmp = {}

        self.__initlization()

    def __initlization(self):
        for i in range(1, self.n_layers, 1):
            n_cur = self.layer_list[i-1]
            n_next = self.layer_list[i]
            self.weights["W" + str(i)] = np.random.normal(size=(n_next, n_cur))
            self.bias["b" + str(i)] = np.random.normal(size=(n_next, 1))
            if self.solver == "mgd" or self.solver == "RMSprop":
                self.Vw["VW" + str(i)] = np.zeros((n_next, n_cur))
                self.Vb["Vb" + str(i)] = np.zeros((n_next, 1))

    def __compute_loss(self, Y, Y_):
        m = Y.shape[1]
        # 损失函数中的sigmoid函数中的指数函数溢出，很可能learning_rate太大，
        # 可以把learning_rate调成非常小，如1e-8

        loss = -(Y * np.log(Y_) + (1 - Y) * np.log(1 - Y_)).sum()

        # def safe_log(x): return np.clip(x, 1.e-100, 1.e+100)
        # loss = -(Y * safe_log(Y_) + (1 - Y) * safe_log(1 - Y_)).sum()

        if abs(self.C) > 1e-8:
            for i in range(1, self.n_layers, 1):
                W = self.weights["W" + str(i)]
                loss += 0.5 * self.C * W.sum()
        loss /= m
        return loss

    def __forward(self, X):
        self.tmp["A1"] = X
        for i in range(1, self.n_layers, 1):
            # 1,...,L - 1
            A = self.tmp["A" + str(i)]
            W = self.weights["W" + str(i)]
            b = self.bias["b" + str(i)]
            self.tmp["A" + str(i + 1)] = self.__sigmoid(W @ A + b)

    def __backward(self, Y):
        m = Y.shape[1]
        delta = self.tmp["A" + str(self.n_layers)] - Y
        for i in range(self.n_layers - 1, 1, -1):
            # L-1,.,2
            W = self.weights["W" + str(i)]
            b = self.bias["b" + str(i)]
            A = self.tmp["A" + str(i)]
            tmp = W.T @ delta * A * (1 - A)
            DW = (delta @ A.T + self.C * W) / m
            Db = delta.sum(axis=1, keepdims=True) / m
            self.__update(W, b, DW, Db, i)

            delta = tmp

    def __update(self, W, b, DW, Db, i):
        if self.solver == "mgd":
            VW = self.Vw["VW" + str(i)]
            Vb = self.Vb["Vb" + str(i)]

            VW = self.beta1 * VW + (1 - self.beta1) * DW
            Vb = self.beta1 * Vb + (1 - self.beta1) * Db

            self.Vw["VW" + str(i)] = VW
            self.Vb["Vb" + str(i)] = Vb

            self.weights["W" + str(i)] = W - self.learning_rate * VW
            self.bias["b" + str(i)] = b - self.learning_rate * Vb

        if self.solver == "RMSprop":
            VW = self.Vw["VW" + str(i)]
            Vb = self.Vb["Vb" + str(i)]

            VW = self.beta2 * VW + (1 - self.beta2) * (DW ** 2)
            Vb = self.beta2 * Vb + (1 - self.beta2) * (Vb ** 2)

            self.Vw["VW" + str(i)] = VW
            self.Vb["Vb" + str(i)] = Vb

            self.weights["W" + str(i)] = W - self.learning_rate * \
                (DW / (np.sqrt(VW) + self.epslion))
            self.bias["b" + str(i)] = b - self.learning_rate * \
                (Db / (np.sqrt(Vb) + self.epslion))

        if self.solver == "gd":
            self.weights["W" + str(i)] = W - self.learning_rate * DW
            self.bias["b" + str(i)] = b - self.learning_rate * Db

    def fit(self, X, Y, batch_size=None):

        if not batch_size:
            it = 0
            while it < self.max_iter:
                self.__forward(X)

                self.loss.append(self.__compute_loss(
                    Y, self.tmp["A" + str(self.n_layers)]))

                self.__backward(Y)

                it += 1
        else:
            it = 0
            m = Y.shape[1]
            while it < self.max_iter:

                index = sample_without_replacement(
                    n_population=m, n_samples=batch_size)

                self.__forward(X[:, index])
                self.loss.append(self.__compute_loss(
                    Y[:, index], self.tmp["A" + str(self.n_layers)]))
                self.__backward(Y[:, index])

                it += 1

        # for i in range(1, self.n_layers + 1, 1):
        #     del self.tmp["A" + str(i)]
        self.tmp = {}

        return self

    def predict(self, X):
        A = X
        for i in range(1, self.n_layers, 1):
            W = self.weights["W" + str(i)]
            b = self.bias["b" + str(i)]
            Z = W @ A + b
            A = self.__sigmoid(Z)
        return A

    def __sigmoid(self, z):
        # def safe_sigmoid(x):
        #     if x >= 0:
        #         return 1.0 / (1.0 + exp(-x))
        #     else:
        #         return exp(x) / (1.0 + exp(x))
        # vsafe_sigmoid = np.vectorize(safe_sigmoid)
        # return vsafe_sigmoid(z)
        return 1.0 / (1.0 + np.exp(-z))

    def __learning_rate_dacay(self, n_iter):
        if "SLR" == self.lr_decay:
            v = n_iter / self.max_iter
            for i in range(0, 10):
                if v >= i * 0.1 and v < (i+1) * 0.1:
                    self.learning_rate = self.learning_rate_init / 2 ** i
                    break
            return

        if "ELR" == self.lr_decay:
            v = n_iter / self.max_iter
            self.learning_rate = self.learning_rate_init * exp(- v / (1 - v))
            return

        if "MSLR" == self.lr_decay:
            pass
        if "CALR" == self.lr_decay:
            pass
