from nn import NerualNetwork
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


def test1():
    X, y = datasets.load_iris(return_X_y=True)
    Y = LabelBinarizer().fit_transform(y)
    X = X.T
    Y = Y.T

    X = StandardScaler().fit_transform(X)
    # X = MinMaxScaler().fit_transform(X)

    n_iter = 500
    layer_list = [4, 10, 3]
    batch_size = 8

    clf1 = NerualNetwork(layer_list=layer_list, solver="gd",
                         learning_rate_init=0.01, max_iter=n_iter, C=0)
    clf1.fit(X, Y, batch_size=batch_size)

    clf2 = NerualNetwork(layer_list=layer_list, solver="mgd",
                         learning_rate_init=0.01, beta1=0.9, max_iter=n_iter, C=0)
    clf2.fit(X, Y, batch_size=batch_size)

    # 损失函数中的sigmoid函数中的指数函数溢出，很可能learning_rate太大，
    # 可以把learning_rate调成非常小，如1e-8
    clf3 = NerualNetwork(layer_list=layer_list, solver="RMSprop",
                         learning_rate_init=1e-9, beta2=0.9999, max_iter=n_iter, C=0)
    clf3.fit(X, Y, batch_size=batch_size)

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(5, 4))
    plt.plot([i for i in range(len(clf1.loss))], clf1.loss, label="gd")
    plt.plot([i for i in range(len(clf2.loss))], clf2.loss, label="mgd")
    plt.plot([i for i in range(len(clf3.loss))], clf3.loss, label="RMSprop")
    plt.legend()
    plt.xlabel("No. of iteration")
    plt.ylabel("Cost")
    plt.show()
    plt.savefig("bdg_and_mbgd.png", dpi=800)


def test2():

    n_iter = 200

    X, y = datasets.load_iris(return_X_y=True)
    Y = LabelBinarizer().fit_transform(y)
    X = X.T
    Y = Y.T

    plt.figure(figsize=(5, 4))
    batch_size_list = [i * 20 for i in range(1, 7)]
    for batch_size in batch_size_list:
        clf = NerualNetwork(layer_list=[4, 10, 3], max_iter=n_iter, C=0)
        clf.fit(X, Y, batch_size=batch_size)
        plt.plot([i for i in range(len(clf.loss))],
                 clf.loss, label=str(batch_size) + "- bgd")
    plt.legend()
    plt.xlabel("No. of iteration")
    plt.ylabel("Cost")
    plt.grid()
    plt.show()
    plt.savefig("mbgd.png", dpi=800)


def test3():
    X, y = datasets.load_wine(return_X_y=True)
    Y = LabelBinarizer().fit_transform(y)
    X = X.T
    Y = Y.T

    layer_list = [X.shape[0], 10, 10, Y.shape[0]]
    n_iter = 250
    batch_size = 32

    clf1 = NerualNetwork(
        solver="mgd", layer_list=layer_list, max_iter=n_iter, C=0)
    clf1.fit(X, Y, batch_size=batch_size)

    clf2 = NerualNetwork(
        solver="gd", layer_list=layer_list, max_iter=n_iter, C=0)
    clf2.fit(X, Y, batch_size=batch_size)

    X = StandardScaler().fit_transform(X)
    clf3 = NerualNetwork(
        solver="mgd", layer_list=layer_list, max_iter=n_iter, C=0)
    clf3.fit(X, Y, batch_size=batch_size)

    clf4 = NerualNetwork(
        solver="gd", layer_list=layer_list, max_iter=n_iter, C=0)
    clf4.fit(X, Y, batch_size=batch_size)

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(5, 4))
    plt.plot([i for i in range(len(clf1.loss))], clf1.loss, label="mgd")
    plt.plot([i for i in range(len(clf2.loss))], clf2.loss, label="gd")
    plt.plot([i for i in range(len(clf3.loss))],
             clf3.loss, label="mgd-standard")
    plt.plot([i for i in range(len(clf4.loss))],
             clf4.loss, label="mgd-standard")
    plt.legend()
    plt.xlabel("No. of iteration")
    plt.ylabel("Cost")
    # plt.show()
    plt.savefig("gd_and_mgd.png", dpi=800)


def test4():
    X, y = datasets.load_wine(return_X_y=True)
    Y = LabelBinarizer().fit_transform(y)
    X = X.T
    Y = Y.T
    X = StandardScaler().fit_transform(X)

    layer_list = [X.shape[0], 10, 10, Y.shape[0]]
    n_iter = 200
    batch_size = 16

    clf1 = NerualNetwork(solver="gd", learning_rate_init=0.01,
                         lr_decay="SLR", layer_list=layer_list, max_iter=n_iter, C=0)
    clf1.fit(X, Y, batch_size=batch_size)

    clf2 = NerualNetwork(solver="gd", learning_rate_init=0.01,
                         layer_list=layer_list, max_iter=n_iter, C=0)
    clf2.fit(X, Y, batch_size=batch_size)

    clf3 = NerualNetwork(solver="gd", learning_rate_init=0.01,
                         lr_decay="ELR", layer_list=layer_list, max_iter=n_iter, C=0)
    clf3.fit(X, Y, batch_size=batch_size)

    # clf4 = NerualNetwork(
    #     solver="gd", layer_list=layer_list, max_iter=n_iter, C=0)
    # clf4.fit(X, Y, batch_size=batch_size)

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(5, 4))
    plt.plot([i for i in range(len(clf1.loss))], clf1.loss, label="gd-slr")
    plt.plot([i for i in range(len(clf2.loss))], clf2.loss, label="gd")
    plt.plot([i for i in range(len(clf3.loss))],clf3.loss, label="gd-elr")
    # plt.plot([i for i in range(len(clf4.loss))],
    #          clf4.loss, label="mgd-standard")
    plt.legend()
    plt.xlabel("No. of iteration")
    plt.ylabel("Cost")
    plt.show()
    plt.savefig("lr_decay.png", dpi=800)


if __name__ == "__main__":
    # test3()
    # test1()
    test4()
