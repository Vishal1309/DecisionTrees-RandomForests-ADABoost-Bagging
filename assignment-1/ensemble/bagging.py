# BAGGING.py

# from tree.base import DecisionTree
import numpy as np
import pandas as pd
from sklearn import tree as sktree
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode
from sklearn.tree import DecisionTreeClassifier
import os

class BaggingClassifier():
    def __init__(self, base_estimator = DecisionTreeClassifier, n_estimators=100, max_depth=100, criterion="entropy"):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator # SKtree
        self.max_depth = max_depth 
        self.trees = []
        self.samples_xy = []
        self.criterion = criterion
        self.n_estimators = n_estimators # number of iterations/ estimators for which we will run bagging.

        pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        # getting data samples and learning models for all those samples:
        for iter in range(self.n_estimators):
            curr_X = X.sample(frac=1, axis='rows', replace=True) # sampling with replacement
            curr_y = y[curr_X.index]

            curr_X = curr_X.reset_index(drop=True) # maininting new indices
            curr_y = curr_y.reset_index(drop=True)

            tree = self.base_estimator(criterion=self.criterion) # learning new model for current sample
            tree.fit(curr_X, curr_y) # fitting in sktree

            self.trees.append(tree) # storing the tree corresponding data publicly
            self.samples_xy.append([curr_X, curr_y])
        pass

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Predicted = None
        i = 0
        for tree in self.trees:
            if Predicted is not None:
                Predicted[i] = tree.predict(X)
            else:
                Predicted = pd.Series(tree.predict(X)).to_frame()
            i += 1
#         print(Predicted)
#         print(Predicted.mode(axis = 1))
        best_prediction = Predicted.mode(axis = 1)
#         print(best_prediction[0])
        return best_prediction[0]
        pass

    def plot(self, X, y):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        color = ["y", "r", "b"]
        Z_ls = []
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X[0].min(), X[0].max()
        x_range = x_max-x_min
        y_min, y_max = X[1].min(), X[1].max()
        y_range = y_max-y_min
#         print(x_range)
#         print(y_range)

        # plotting surfaces for all the sampling iterations:
        for i, tree in enumerate(self.trees):
            X_tree, y_tree = self.samples_xy[i]

            xx, yy = np.meshgrid(np.arange(x_min-0.5, x_max+0.5, (x_range)/50),
                                 np.arange(y_min-0.5, y_max+0.5, (y_range)/50))
            
            ax1[i].set_xlabel("X1")
            ax1[i].set_ylabel("X2")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Z_ls.append(Z)
            c_surf = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.YlOrBr)
            fig1.colorbar(c_surf, ax=ax1[i])

            for y_label in y.unique():
                idx = y_tree == y_label
                id = list(y_tree.cat.categories).index(y_tree[idx].iloc[0])
                ax1[i].scatter(X_tree.loc[idx, 0], X_tree.loc[idx, 1], c=color[id],
                               cmap=plt.cm.YlOrBr, s=30,
                               label="Class_Type: "+str(y_label))
            ax1[i].legend()
            ax1[i].set_title("Decision Tree Surface: {}".format(i + 1))
            

        # plottinh common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        Z_ls = np.array(Z_ls)
        Common_surf, bleh = weighted_mode(Z_ls, np.ones(Z_ls.shape))
        c_surf = ax2.contourf(xx, yy, Z, cmap=plt.cm.YlOrBr)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X.loc[idx, 0], X.loc[idx, 1], c=color[id],
                        cmap=plt.cm.YlOrBr, s=30,
                        label="Class_Type: "+str(y_label))
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.legend()
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(c_surf, ax=ax2)

        fig1.savefig("Q6_fig1.png")
        fig2.savefig("Q6_fig2.png")
        return fig1, fig2
        pass
