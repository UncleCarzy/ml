import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import queue
import random
from sklearn.cluster import DBSCAN as DBS
import time


class DBSCAN:

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        """[summary]

        Keyword Arguments:
            eps {float} -- [The maximum distance between two samples for one to be considered as in the neighborhood of the other] (default: {0.5})
            min_samples {int} -- [The number of samples in a neighborhood for a point to be considered as a core point] (default: {5})
            metric {str} -- [description] (default: {"euclidean"})
        """
        self.eps_square = eps ** 2
        self.min_samples = min_samples

    def fit_predict(self, X):
        """[Perform DBSCAN clustering from features or distance matrix]

        Arguments:
            X {[array]} -- [shape = (n_samples, n_features)]
        """
        n_samples = X.shape[0]
        core_set = set()
        neighbor = [set() for i in range(n_samples)]
        for j in range(n_samples):
            diff = X - X[j, :]
            dist_square = (diff * diff).sum(axis=1)
            for i in range(n_samples):
                if dist_square[i] < self.eps_square:
                    neighbor[j].add(i)
            if len(neighbor[j]) >= self.min_samples:
                core_set.add(j)
        clusters = []
        k = 0
        unvistied = set([i for i in range(n_samples)])
        while len(core_set) > 0:
            unvistied_old = set(unvistied)
            rdx = unvistied.pop()  # 应该是随机选一个的，这里默认选第一个
            Q = queue.SimpleQueue()
            Q.put(rdx)
            while not Q.empty():
                q = Q.get()
                if len(neighbor[q]) >= self.min_samples:
                    delta = neighbor[q].intersection(unvistied)
                    for item in delta:
                        Q.put(item)
                        unvistied.discard(item)
            clusters.append(unvistied_old.difference(unvistied))
            core_set.difference_update(clusters[k])
            k += 1
        y = np.zeros(n_samples, dtype='int')
        for i in range(k):
            for item in clusters[i]:
                y[item] = i
        return y


if __name__ == "__main__":
    X, y = datasets.load_iris(return_X_y=True)
    db = DBS(eps=0.4, min_samples=10)
    # db = DBSCAN(eps=0.25, min_samples=10)
    tag = db.fit_predict(X[:, [0, 3]])
    print(tag)
    print(y)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 3], c=tag)
    plt.grid(linestyle="--")
    plt.savefig("dbscan.png", dpi=800)
