# ADABoost.py

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as sktree
import os


class AdaBoostClassifier():
    # Optional Arguments: Type of estimator
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=5, criterion="entropy"):
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.max_depth = 1 # decision stump
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.alphas = []
        self.d_stumps = []

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        weight_dp = np.ones(len(y))/len(y) # weight for all datapoints
        for n in range(self.n_estimators):
            tree = self.base_estimator(criterion=self.criterion, max_depth=1)
            tree.fit(X, y, sample_weight=weight_dp)
            y_hat = pd.Series(tree.predict(X))
            right_ids, wrong_ids = y_hat == y, y_hat != y
            sum_wdp = np.sum(weight_dp)
            error = np.sum(weight_dp[wrong_ids])/sum_wdp
            alpha_m = 0.5*np.log((1-error)/error) # by formula of alpha

            # updating the weights
            weight_dp[wrong_ids] *= np.exp(alpha_m)
            weight_dp[right_ids] *= np.exp(-alpha_m)
            weight_dp /= np.sum(weight_dp) # normalizing for all datapoints

            self.d_stumps.append(tree) # storing the decision trees obtained so far
            self.alphas.append(alpha_m)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Pred = 0
        for i, (alpha_m, tree) in tqdm(enumerate(zip(self.alphas, self.d_stumps))): # iterating over the already learnt d_stumps
#             print(alpha_m)
#             print(tree)
            if i != 0:
                Pred += pd.Series(tree.predict(X))*alpha_m
            else:
                Pred = pd.Series(tree.predict(X))
                Pred *= alpha_m # first prediction
        Predicted = Pred.apply(np.sign) # [-1, +1] : [neg, pos]
        return Predicted

    def plot(self, X, y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """

        assert(len(list(X.columns)) == 2)
        color = ["y", "r", "b"]
        
        fig1, ax1 = plt.subplots(1, len(self.d_stumps), figsize=(5*len(self.d_stumps), 4))

        x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
        x_range = x_max-x_min
        y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
        y_range = y_max-y_min
        H = (x_range) / 50
        H_ = (y_range) / 50
        y_hat_ls = None
        # for every Decision Stump, make a surface plot
        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.d_stumps)):
            print("Adaboost classifier {}:".format(i+1))
            print(sktree.export_text(tree)) # sktree inbuilt tree print
            attr_x, attr_y = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H_))
            ax1[i].set_xlabel("X1")
            ax1[i].set_ylabel("X2")
            y_hat = tree.predict(np.c_[attr_x.ravel(), attr_y.ravel()])
            y_hat = y_hat.reshape(attr_x.shape)
            if y_hat_ls is not None:
                y_hat_ls += alpha_m*y_hat
            else:
                y_hat_ls = alpha_m*y_hat
            c_surf = ax1[i].contourf(attr_x, attr_y, y_hat, cmap=plt.cm.YlOrBr)
            fig1.colorbar(c_surf, ax=ax1[i])
            for y_title in y.unique():
                idx = y == y_title
                id = list(y.cat.categories).index(y[idx].iloc[0])
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id], cmap=plt.cm.YlOrBr, s=30, label="Class_Type: {}".format(y_title))
            ax1[i].legend()
            ax1[i].set_title("Adaboost classifier {}: ".format(i + 1))
            

        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
#         y_hat_ls = np.array(y_hat_ls)
#         print(y_hat_ls)
        Common_surf = np.sign(y_hat_ls)
        c_surf = ax2.contourf(attr_x, attr_y, y_hat, cmap=plt.cm.YlOrBr)
        for y_title in y.unique():
            idx = y == y_title
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].loc[:,0], X[idx].iloc[:, 1], c=color[id], cmap=plt.cm.YlOrBr, s=30, label="Class_Type: {}".format(y_title))
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.set_title("Common Adaboost surface: ")
        ax2.legend()
        fig2.colorbar(c_surf, ax=ax2)

        # Saving Figures
#         fig1.savefig("Q5_fig1.png")
#         fig2.savefig("Q5_fig2.png")
        return fig1, fig2

# np.random.seed(42)

# N = 30
# P = 2
# NUM_OP_CLASSES = 2
# n_estimators = 3
# X = pd.DataFrame(np.abs(np.random.randn(N, P)))
# y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

# criteria = 'information_gain'
# tree = DecisionTreeClassifier
# Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
# Classifier_AB.fit(X, y)
# y_hat = Classifier_AB.predict(X)
# [fig1, fig2] = Classifier_AB.plot(X, y)
# print('Criteria :', criteria)
# print('Accuracy: ', accuracy(y_hat, y))
# for cls in y.unique():
#     print('Precision: ', precision(y_hat, y, cls))
#     print('Recall: ', recall(y_hat, y, cls))