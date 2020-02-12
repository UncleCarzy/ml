import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer, Normalizer


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NerualNetwork:

    def __init__(self, layer_list=[], learning_rate=0.01, alpha=1.0, max_iter=200):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.layer_list = layer_list
        self.n_layers = len(layer_list)
        self.n_outputs = layer_list[-1]
        self.loss = []
        self.weights = {}
        self.bias = {}
        self.tmp = {}

        for i in range(1, self.n_layers, 1):
            n_cur = self.layer_list[i-1]
            n_next = self.layer_list[i]
            self.weights["W" + str(i)] = np.random.normal(size=(n_next, n_cur))
            self.bias["b" + str(i)] = np.random.normal(size=(n_next, 1))

    def __compute_loss(self, Y, Y_):
        m = Y.shape[1]
        loss = None
        loss = (- Y * np.log(Y_) - (1 - Y) * np.log(1 - Y_)).sum()
        for i in range(1, self.n_layers, 1):
            W = self.weights["W" + str(i)]
            loss += 0.5 * self.alpha * W.sum()
        loss /= m
        return loss

    def fit(self, X, Y):
        m = X.shape[1]
        it = 0
        while it < self.max_iter:
            self.tmp["A1"] = X
            for i in range(1, self.n_layers, 1):
                # 1,...,L - 1
                A = self.tmp["A" + str(i)]
                W = self.weights["W" + str(i)]
                b = self.bias["b" + str(i)]
                self.tmp["A" + str(i + 1)] = sigmoid(W @ A + b)
            self.loss.append(self.__compute_loss(
                Y, self.tmp["A" + str(self.n_layers)]))
            delta = self.tmp["A" + str(self.n_layers)] - Y
            for i in range(self.n_layers - 1, 1, -1):
                # L-1,.,2
                W = self.weights["W" + str(i)]
                b = self.bias["b" + str(i)]
                A = self.tmp["A" + str(i)]
                tmp = W.T @ delta * A * (1 - A)
                DW = (delta @ A.T + self.alpha * W) / m
                Db = delta.sum(axis=1, keepdims=True) / m
                self.weights["W" + str(i)] = W - self.learning_rate * DW
                self.bias["b" + str(i)] = b - self.learning_rate * Db
                delta = tmp
            it += 1

        for i in range(1, self.n_layers + 1, 1):
            del self.tmp["A" + str(i)]

        return self

    def predict(self, X):
        A = X
        for i in range(1, self.n_layers, 1):
            W = self.weights["W" + str(i)]
            b = self.bias["b" + str(i)]
            Z = W @ A + b
            A = sigmoid(Z)
        return A
