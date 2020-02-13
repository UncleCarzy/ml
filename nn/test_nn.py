from nn import NerualNetwork
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import matplotlib.pyplot as plt


def test1():
    X, y = datasets.load_iris(return_X_y=True)
    Y = LabelBinarizer().fit_transform(y)
    X = X.T
    Y = Y.T

    n_iter = 100

    clf1 = NerualNetwork(layer_list=[4, 10, 3], max_iter=n_iter, C=0)
    clf1.fit(X, Y, batch_size=10)

    clf2 = NerualNetwork(layer_list=[4, 10, 3], max_iter=n_iter, C=0)
    clf2.fit(X, Y)

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(5, 4))
    plt.plot([i for i in range(len(clf1.loss))], clf1.loss, label="mbgd")
    plt.plot([i for i in range(len(clf2.loss))], clf2.loss, label="bgd")
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
    plt.show()
    plt.savefig("gd_and_mgd.png", dpi=800)


if __name__ == "__main__":
    test3()
    pass
