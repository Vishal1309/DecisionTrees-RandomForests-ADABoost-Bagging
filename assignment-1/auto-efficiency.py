# auto-efficiency.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read real-estate data set
# ...
# 
Xy = pd.read_csv("C:/Users/Vishal J. Soni/Desktop/Sem 6/Machine Learning/assignment-1-Vishal1309/assignment-1/auto-mpg.csv")
# print(Xy.head(5))
# print(Xy.hp.mean()) ERROR
# some values of hp are "?"
Xy.hp = Xy.hp.str.replace('?', 'NaN').astype(float)
Xy.hp.fillna(Xy.hp.mean(), inplace=True)
d_hp = Xy.hp
Xy.hp = d_hp.astype(int)
# print(Xy.hp.mean())
# since carname is not a useful attr, deleting that column
Xy = Xy.drop('carname', axis=1)

m,n = Xy.shape
# print(Xy)
for i, attr in enumerate(Xy):
    if i == 0:
        continue
    else:
        Xy.rename(columns={str(attr):i-1}, inplace=True)
# print(Xy)

# Preprocessing
shuffled = Xy.sample(frac=1).reset_index(drop=True)
X = (shuffled.iloc[:, 1:]).squeeze()
y = (shuffled.iloc[:, 0:1]).T.squeeze()
# print(X)
# print(y)
# split = 70% of data
split = int(0.7*len(y))
X_train=pd.DataFrame(X.iloc[:split], dtype=np.float64)
X_test=pd.DataFrame(X.iloc[split:], dtype=np.float64)
y_train=pd.Series(y[:split], dtype=np.float64, name=None)
y_test=pd.Series(y[split:], dtype=np.float64, name=None)
# print(X_train)
max_depth = 5 
# # training on my Decision Tree:
tree = DecisionTree(max_depth=max_depth)
tree.fit(X_train, y_train)
tree.plot()
y_hat1 = pd.Series(tree.predict(X_test))
y_test_1 = y_test.reset_index(drop=True)
RMSE_Mine = rmse(y_hat1, y_test_1)
MAE_Mine = mae(y_hat1, y_test_1)
print('MyTree RMSE: {}'.format(RMSE_Mine))
print('MyTree MAE: {}'.format(MAE_Mine))

# sktree training
tree_sk = DecisionTreeRegressor(random_state=0)
tree_sk.fit(X_train, y_train)
y_hat = pd.Series(tree_sk.predict(X_test))
y_test_ = y_test.reset_index(drop=True)
RMSE_sk = rmse(y_hat, y_test_)
MAE_sk = mae(y_hat, y_test_)
print('SkTree  RMSE: {}'.format(RMSE_sk))
print('SkTree  MAE: {}'.format(MAE_sk))
if RMSE_Mine <= RMSE_sk:
    print("RMSE of my tree is better than that of the Sktree")
else:
    print("RMSE of SkTree is better than that of MyTree")
if MAE_Mine <= MAE_sk:
    print("MAE of MyTree is better than that of SkTree")
else:
    print("MAR of SkTree is better than that of MyTree")
