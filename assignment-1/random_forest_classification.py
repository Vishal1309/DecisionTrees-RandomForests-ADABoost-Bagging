import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

###Write code here

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)


X=pd.DataFrame(X)
y=pd.Series(y)
Xy=pd.concat([X,y], axis=1)
pre_Xy = Xy.sample(frac=1).reset_index(drop=True)

X = pre_Xy.iloc[:, :-1].squeeze()
y = (pre_Xy.iloc[:, -1:]).T.squeeze()
# print(X)


X_train = X.loc[:0.6*(len(y))-1]
y_train = y.loc[:0.6*(len(y))-1]
X_test = X.loc[0.6*(len(y)):].reset_index(drop=True)
y_test = y.loc[0.6*(len(y)):].reset_index(drop=True)
# print(X_test)

for criteria in ["entropy", "gini"]:
    Classifier_RF = RandomForestClassifier(7, criterion=criteria, attr_lim=2)
    Classifier_RF.fit(X_train, y_train)
    y_hat = Classifier_RF.predict(X_test)
    # Classifier_RF.plot()
    # print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))
