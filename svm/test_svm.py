from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


X, y = datasets.load_wine(return_X_y=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
clf1 = SVC(kernel="linear")
clf1.fit(Xtrain, ytrain)
ytrain_ = clf1.predict(Xtrain)
ytest_ = clf1.predict(Xtest)
print("linear kernel train acc: %.3f" % accuracy_score(ytrain, ytrain_))
print("linear kernel test acc: %.3f" % accuracy_score(ytest, ytest_))

clf2 = SVC(kernel="rbf")
clf2.fit(Xtrain, ytrain)
ytrain_ = clf2.predict(Xtrain)
ytest_ = clf2.predict(Xtest)
print("rbf kernel train acc: %.3f" % accuracy_score(ytrain, ytrain_))
print("rbf kernel test acc: %.3f" % accuracy_score(ytest, ytest_))
