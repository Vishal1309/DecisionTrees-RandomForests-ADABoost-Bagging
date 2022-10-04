import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read dataset
# ...
# 

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

X_train = pd.DataFrame(X[0:70,:])
# print(X_train)
y_train = pd.Series(y[0:70], dtype = "category")
# print(y_train)
X_test = pd.DataFrame(X[70:100,:])
y_test = pd.Series(y[70:100], dtype = "category")

for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))
        
# 1a done here. ------------------------------------------------------------------------------------------

from sklearn.model_selection import KFold

def get_opt_depth(X, y, folds = 5, depths = [3, 4, 5, 6]):
    kf = KFold(n_splits = folds)
    store = kf
    kf.get_n_splits(X)
    avg_Accuracy = {"information_gain":0, "gini_index":0}
    a1, a2 = 0, 0
    for train_id, test_id in kf.split(X):
        X_train = pd.DataFrame(X[train_id])
        y_train = pd.Series(y[train_id], dtype = "category")
        X_test = pd.DataFrame(X[test_id])
        y_test = pd.Series(y[test_id], dtype = "category")
#         print(X_train)
#         print(y_train)
#         print(X_test)
#         print(y_test)
        for criteria in ['information_gain', 'gini_index']:
            tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
            tree.fit(X_train, y_train)
            y_hat = tree.predict(X_test)
#             tree.plot()
            print('Criteria :', criteria)
            print('Accuracy: ', accuracy(y_hat, y_test))
            avg_Accuracy[criteria] += accuracy(y_hat, y_test)
    for criteria in ['information_gain', 'gini_index']:
        avg_Accuracy[criteria] /= folds
    print("KFold avg accuracy on information gain: {:.2f} and on gini_index = {:.2f}".format(avg_Accuracy["information_gain"], avg_Accuracy["gini_index"]))
    f = 0
    kf = store
    kf.get_n_splits(X)
    for train_id, test_id in kf.split(X):
        X_train = X[train_id]
        y_train = y[train_id]
        best_Accuracy = {"information_gain":0, "gini_index":0}
        best_depth = {"information_gain":0, "gini_index":0}
        for d in depths:
            kf_val = KFold(n_splits = folds)
            kf_val.get_n_splits(X_train)
            curr_avg_Accuracy = {"information_gain":0, "gini_index":0}
            for train_id_nest, test_id_nest in kf_val.split(X_train):
                X_train_nest = pd.DataFrame(X_train[train_id_nest])
                y_train_nest = pd.Series(y_train[train_id_nest], dtype = "category")
                X_val = pd.DataFrame(X_train[test_id_nest])
                y_val = pd.Series(y_train[test_id_nest], dtype = "category")
                for criteria in ['information_gain', 'gini_index']:
                    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
                    tree.fit(X_train_nest, y_train_nest)
#                     X_val = pd.DataFrame(X_val)
                    y_hat = tree.predict(X_val)
#                     tree.plot()
                    curr_avg_Accuracy[criteria] += accuracy(y_hat, y_val)
            for criteria in ['information_gain', 'gini_index']:
                curr_avg_Accuracy[criteria] /= folds
                if curr_avg_Accuracy[criteria] > best_Accuracy[criteria]:
                    best_Accuracy = curr_avg_Accuracy
                    best_depth[criteria] = d
        f += 1
        print("Optimal Depth for fold {} on information gain = {} and on gini_index = {}".format(f, best_depth["information_gain"], best_depth["gini_index"]))
        
            

get_opt_depth(X, y, folds = 5, depths = list(range(3, 12))) # calling the func to print most optimal depth using 5 fold nested cross validation

