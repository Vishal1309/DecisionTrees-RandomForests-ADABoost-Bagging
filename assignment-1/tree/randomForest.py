# from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import matplotlib.image as mpimg


np.random.seed(42)
class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, attr_lim=3):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.X=None
        self.y=None
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.samples_store=[] # storing the attributes sampled at every iteration
        self.attr_lim=attr_lim # max number of attr at every iteration
        self.trees=[] # storing the trees

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X=X
        self.y=y # will help while plotting
        for n in range(self.n_estimators):
            curr_X=X.sample(np.random.randint(1,self.attr_lim+1), axis='columns') # sampling with num_subsets in range 1, attr_lim

            tree=DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth)
            tree.fit(curr_X,y)
            self.samples_store.append([])
            for col in curr_X.columns:    # store the samples (attributes) of curr_sample
                self.samples_store[n].append(col)
            self.trees.append(tree) # storing the curr_tree
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        for i, (tree,samples_store) in enumerate(zip(self.trees, self.samples_store)):
            if i==0:
                preds=pd.Series(tree.predict(X[samples_store])).to_frame()
            else:
                preds[i]=tree.predict(X[samples_store])
        Predicted=preds.mode(axis=1)[0]  # return the best/mode of the stored predictions
        return Predicted
        pass
    
    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        X = self.X
        y = self.y
        plt.figure(figsize=(8, 8))
        for i, treee in enumerate(self.trees):
            sktree.plot_tree(treee)
            plt.savefig('PLOTS/Q7_tree_{}.png'.format(i + 1))
        # surface plotting code similar to that of bagging...
        
        pass




class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.X=None
        self.y=None
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.samples_store=[] # storing the attributes sampled at every iteration
        self.attr_lim=attr_lim # max number of attr at every iteration
        self.trees=[] # storing the trees

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X=X
        self.y=y # will help while plotting
        for n in range(self.n_estimators):
            curr_X=X.sample(np.random.randint(1,self.attr_lim+1), axis='columns') # sampling with num_subsets in range 1, attr_lim

            tree=DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth)
            tree.fit(curr_X,y)
            self.samples_store.append([])
            for col in curr_X.columns:    # store the samples (attributes) of curr_sample
                self.samples_store[n].append(col)
            self.trees.append(tree) # storing the curr_tree
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        for i, (tree,samples_store) in enumerate(zip(self.trees, self.samples_store)):
            if i==0:
                preds=pd.Series(tree.predict(X[samples_store])).to_frame()
            else:
                preds[i]=tree.predict(X[samples_store])
        Predicted=preds.mean(axis=1)[0]  # return the best/mode of the stored predictions
        return Predicted
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
