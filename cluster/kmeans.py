import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class KMeans:
    """[K-Means clustering.]
    """

    def __init__(self, n_clusters=4, max_iter=300, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cost = []
        # array of shape (n_clusters, n_features)
        self.cluster_centers = None

    def fit(self, X):
        """[Compute K-means clustering.]

        Arguments:
            X {[array]} -- [shape = (n_examples, n_features)]
        """
        n_examples = X.shape[0]
        # initialize
        random_idx = np.random.permutation(
            n_examples)[:self.n_clusters].tolist()
        self.cluster_centers = X[random_idx, :]
        # record the idx of examples belonging to the cluster centers
        tag = [[] for i in range(self.n_clusters)]
        # record the cost
        self.cost.clear()
        pre_cost = 0.0
        cost = 0.0

        iter = 0
        while (iter < self.max_iter):
            cost_arr = np.zeros(self.n_clusters, dtype='float')
            count = np.zeros(self.n_clusters, dtype='int')
            for i in tag:
                i.clear()

            for i in range(n_examples):
                diff = self.cluster_centers - X[i, :]
                dist = (diff * diff).sum(axis=1)
                min_idx = dist.argmin()
                tag[min_idx].append(i)
                cost_arr[min_idx] += dist[min_idx]
                count[min_idx] += 1

            # update the cluster centers, ignore the centers without examples belonging to them
            for i in range(self.n_clusters):
                if count[i] != 0:
                    self.cluster_centers[i, :] = X[tag[i], :].mean(
                        axis=0, keepdims=True)

            # compute and store the cost
            pre_cost = cost
            cost = cost_arr.sum()
            self.cost.append(cost / n_examples)
            #print("iter = %d Cost = %f diff = %f" % (iter, cost, pre_cost - cost))
            if (iter != 0) and (pre_cost - cost < self.tol):
                break
            iter += 1

    def predict(self, X):
        """[Predict the closest cluster each example in X belongs to]

        Arguments:
            X {[array]} -- [shape = (n_samples, n_features)]
        """
        n_samples = X.shape[0]
        tag = np.zeros(n_samples, dtype='int')
        for i in range(n_samples):
            diff = self.cluster_centers - X[i, :]
            dist = (diff * diff).sum(axis=1)
            min_idx = dist.argmin()
            tag[i] = min_idx
        return tag


if __name__ == "__main__":
    X, y = datasets.load_iris(return_X_y=True)
    knn = KMeans(n_clusters=3)
    knn.fit(X[:, [0, 3]])
    tag = knn.predict(X[:, [0, 3]])
    print((y == tag).mean())

    fig, axes = plt.subplots(nrows = 1,ncols = 2)
    axes[0].scatter(X[:, 0], X[:, 3], c=y)
    axes[0].grid(linestyle = "--")
    axes[1].scatter(X[:, 0], X[:, 3], c=tag)
    axes[1].grid(linestyle = "--")
    # fig.savefig("scatter.png",dpi=800)
    fig.show()  

    plt.figure()
    plt.plot([i for i in range(len(knn.cost))], knn.cost)
    plt.grid()
    plt.xlabel("No. of iteration")
    plt.ylabel("Cost")
    plt.show()
    # plt.savefig("Cost line.png", dpi=800)