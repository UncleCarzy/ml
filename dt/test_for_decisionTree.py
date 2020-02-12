import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from decisiontree import DecisionTree
from treePloter import plot_decisionTree


def test_pre_pruning():
    X, y = datasets.load_wine(return_X_y=True)
    X = PCA(n_components=4).fit_transform(X)
    # df = pd.read_csv("watermaleon.csv", encoding="ansi")
    # X = df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']].values
    # y = df["好瓜"].values

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7)

    clf1 = DecisionTree(pruning=None)
    clf1.fit(Xtrain, ytrain, Xtest, ytest)
    ytrain_pred = clf1.predict(Xtrain)
    ytest_pred = clf1.predict(Xtest)
    print("acc before pre_pruning")
    print("train acc: %.3f " % ((ytrain_pred == ytrain).mean() * 100.0))
    print("test acc: %.3f" % ((ytest_pred == ytest).mean() * 100.0))
    plot_decisionTree(clf1.my_tree)
    plt.savefig("不剪枝.png")

    clf2 = DecisionTree(pruning="pre_pruning")
    clf2.fit(Xtrain, ytrain, Xtest, ytest)
    ytrain_pred = clf2.predict(Xtrain)
    ytest_pred = clf2.predict(Xtest)
    print("acc after pre_pruning")
    print("train acc: %.3f " % ((ytrain_pred == ytrain).mean() * 100.0))
    print("test acc: %.3f" % ((ytest_pred == ytest).mean() * 100.0))
    plot_decisionTree(clf2.my_tree)
    plt.savefig("预剪枝.png")

    clf3 = DecisionTree(pruning="post_pruning")
    clf3.fit(Xtrain, ytrain, Xtest, ytest)
    ytrain_pred = clf3.predict(Xtrain)
    ytest_pred = clf3.predict(Xtest)
    print("acc after post_pruning")
    print("train acc: %.3f " % ((ytrain_pred == ytrain).mean() * 100.0))
    print("test acc: %.3f" % ((ytest_pred == ytest).mean() * 100.0))
    plot_decisionTree(clf3.my_tree)
    plt.savefig("后剪枝.png")


if __name__ == "__main__":
    test_pre_pruning()
    pass
