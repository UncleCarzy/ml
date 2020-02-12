import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA as PC


class PCA:
    """[Principal component analysis]
    """

    def __init__(self, n_components=None):
        """[summary]

        Keyword Arguments:
            n_components {[int]} -- [Number of components to keep] (default: {None})
        """
        self.n_components = n_components

        # array, shape = (n_components,)
        self.explained_variance_ = None

        # array, shape = (n_components,)
        self.explained_variance_ratio = None

        # array, shape = (n_compnents, n_features)
        self.components_ = None

        # array, shape = (n_comonents,)
        self.singular_values_ = None

        # array, shape = (n_features,)
        self.mean_ = None

    def fit_transform(self, X):
        """[summary]

        Arguments:
            X {[array]} -- [shape = (n_samples, n_features)]
        """
        # mean normalization
        self.mean_ = X.mean(axis=0, keepdims=True)
        X = X - self.mean_

        # covariance matrix
        sigma = X @ X.T

        # singular value decompsition
        # sigma = u @ s @ vh
        # u 2D-array, s 1D-array, vh 2D-array
        u, s, vh = np.linalg.svd(sigma)
        self.components_ = u[:, :self.n_components]
        self.explained_variance_ = s[:self.n_components]
        self.explained_variance_ratio = self.explained_variance_ / \
            self.explained_variance_.sum()
        self.singular_values_ = s

        return self.components_.T @ X


if __name__ == "__main__":
    X, y = datasets.load_iris(return_X_y=True)
    X = X.T
    pca1 = PCA(3)
    X_ = pca1.fit_transform(X)
    print(pca1.explained_variance_ratio)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_[0, :], X_[1, :], X_[2, :], c=y)
    fig.show()
