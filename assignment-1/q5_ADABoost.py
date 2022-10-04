"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

# from ensemble.ADABoost import AdaBoostClassifier
# from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(X, y)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Classification data set using the entire data set

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

X = pd.DataFrame(X)
y = pd.Series(y)
Xy = pd.concat([X,y], axis=1)
# print(Xy)
pp_Xy = Xy.sample(frac=1, random_state=42).reset_index(drop=True) # pre-processed Xy
# print(pp_Xy)
X = pp_Xy.iloc[:, :-1].squeeze()
y = (pp_Xy.iloc[:, -1:]).T.squeeze()
# print(X)
X_train = X.loc[:0.6*(len(y)) - 1]
y_train = y.loc[:0.6*(len(y))-1]
X_test = X.loc[0.6*(len(y)):].reset_index(drop=True)
y_test = y.loc[0.6*(len(y)):].reset_index(drop=True)
# print(X_test)

tree = AdaBoostClassifier()
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print("Accuracy: {}".format(accuracy(y_hat, y_test)))
for cls in y_test.unique():
    print('Precision for Adaboost: ', precision(y_hat, y_test, cls))
    print('Recall for Adaboost: ', recall(y_hat, y_test, cls))
# [fig1,fig2]=tree.plot(X_test, y_test)

# fig1.savefig("PLOTS/Adaboost_Classifiers.png")
# fig2.savefig("PLOTS/Adaboost_Overall.png")

# comparing our Adaboost function with a Descision Stump
dtree = DecisionTreeClassifier(criterion =  "entropy", max_depth = 1)
dtree.fit(X_train, y_train)
y_hat_dtree = pd.Series(dtree.predict(X_test))
print("Accuracy of a Decision Stump on the dataset: {}".format(accuracy(y_hat_dtree, y_test)))
for cls in y_test.unique():
    print('Precision for Decision Stump: {}'.format(precision(y_hat_dtree, y_test, cls)))
    print('Recall for Decision Stump: {}'.format(recall(y_hat_dtree, y_test, cls)))


