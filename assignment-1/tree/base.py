# BASE.py
#c

"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)


class TreeNode():
    '''
    Nodes for tree, Decision tree stores all the val in form of multiple
    nodes of this class.
    '''

    def __init__(self, split_loc = None, val = None, depth = None):
        self.val = val # value of the TreeNode
        self.split_loc = split_loc # col ( attr) that has to be taken into check (><=)
        self.children = {} # to store child TreeNodes
        self.prob = None # for attr not there
        self.depth = depth # curr_depth
        self.mean = None # when attr is real
        
        
    def __printleafformat__(self, gap = None, val = None, depth = None):
        if val.dtype.name == "str":
            print("|   "*gap + "|--- val = {} Depth = {}".format(val, depth))
        else:
            print("|   "*gap + "|--- val = {:.2f} Depth = {}".format(val, depth))
    
    def __get_sign__(self, child):
        if child == "left":
            return "<"
        else:
            return ">"

    def __print__(self, gap=0):
        '''
        printing the Tree
        i.e, all the nodes and their values in a tree like fashion
        
        '''
        if self.split_loc == None: # Leaf node
            self.__printleafformat__(gap=gap,val=self.val, depth=self.depth)
        else: # non-leaf node
            for child in self.children:
                if self.children[child].prob != None: # classifyable
                    print("|   "*gap + "| ?(X({}) = {}):".format(self.split_loc, child))
                else: # trees needing regression
                    print("|   "*gap+"| ?(X({}) {} {:.2f}):".format(self.split_loc, self.__get_sign__(child), self.mean))
                self.children[child].__print__(gap + 1)

    def get_node_val(self, X, max_depth = np.inf):
        '''
        function to get the value of the node at maximum depth, i.e leaf node where the probability is the maximum
        
        '''
        if self.depth >= max_depth or self.split_loc == None: # leaf reached
            return self.val

        else:
            if self.mean != None: # regressor type
                curr_split = self.split_loc
                if X[curr_split] > self.mean:
                    return self.children["right"].get_node_val(X, max_depth = max_depth)
                else:
                    return self.children["left"].get_node_val(X, max_depth = max_depth)
            else: # classifying into children classes
                curr_split = self.split_loc
                if X[curr_split] in self.children:
                    return self.children[X[curr_split]].get_node_val(X.drop(curr_split), max_depth = max_depth)
                else:
                    max_prob = 0
                    for child in self.children:
                        if child.prob > max_prob:
                            max_prob = child.prob
                    return child.get_node_val(X.drop(self.split_loc), max_depth=max_depth)

class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=10):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion 
        self.max_depth = max_depth
        self.root = None # root node of the TreeNode DecisionTree
        self.attr = None # column names in X
        self.Y_dataType = None # data type of Y to identify whether a classification or regression problem
        self.size_X = None # size (length) of X

    def build_tree(self, X, Y, parent_node, depth=0):
        '''
        build tree func to recursively make the tree
        curr_root: root node currently being passed in the function, whereas depth maintains the level of the tree we are at.
        '''
        if Y.unique().size == 1: # the TreeNode root passed has only one class left
            return TreeNode(val = Y.values[0], depth=depth)

        if len(X.columns) > 0 and depth < self.max_depth and len(list(X.columns)) != sum(list(X.nunique())):
            M_iGain = -np.inf # initializing max information gain
            max_mean = None # initializing maximum mean
            for col in list(X.columns):
                curr_mean = None
                curr_iGain = information_gain(Y, X[col], self.criterion)
                if type(curr_iGain) == tuple:
                    # If attribute selected is range of values
                    curr_mean = curr_iGain[1]
                    curr_iGain = curr_iGain[0]
                if curr_iGain > M_iGain:
                    M_iGain = curr_iGain
                    split_loc = col
                    max_mean = curr_mean
                    
            # now we have the best column that has to be used as the current node so we set the curr_node as this column
            curr_node = TreeNode(split_loc = split_loc) 
            root_col = X[split_loc]
            if root_col.dtype.name != "category":
                # if the root column identified by best info_gain and turns out to have a real distribution
                left_rows = root_col <= max_mean
                right_rows = root_col >= max_mean # splitting the root column in left and right branches

                curr_node.children["left"] = self.build_tree(X[left_rows], Y[left_rows], curr_node, depth=depth+1)
                curr_node.children["right"] = self.build_tree(X[right_rows], Y[right_rows], curr_node, depth=depth+1)
                curr_node.mean = max_mean   
            else:
                # if best column is discrete
                X = X.drop(split_loc, axis=1) # removing this column from the attribute table
                classes_in_root = root_col.groupby(root_col).count() # grouping the same values and with count of different values
                
                for class_type in list(classes_in_root.index):
                    curr_rows = root_col == class_type
                    if curr_rows.sum() > 0:
                        curr_node.children[class_type] = self.build_tree(X[curr_rows], Y[curr_rows], curr_node, depth=depth+1)
                        curr_node.children[class_type].prob = len(X[curr_rows])/self.size_X
                # we've now split the root's children into different classes
                

            if Y.dtype.name != "category": # assigning value to the current roots
                curr_node.val = Y.mean()
            else:
                curr_node.val = Y.mode(dropna=True)[0]

            curr_node.depth = depth # assigning the level at which the current roots lie

            return curr_node
        else:
            # max depth reached or equal values or dataset end reached
            if y.dtype.name != "category": 
                return TreeNode(val = Y.mean(), depth = depth)
            else: # discrete
                return TreeNode(val = Y.mode(dropna=True)[0], depth = depth)

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.Y_dataType = y.dtype
        self.attr = y.name
        self.size_X = len(X)
        self.root = self.build_tree(X, y, None)
        self.root.prob = 1

    def predict(self, X, max_depth=np.inf):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Y = []
        for x in tqdm(X.index):
            Y.append(self.root.get_node_val(X.loc[x], max_depth=max_depth))
        Y_hat = pd.Series(Y, name=self.attr).astype(self.Y_dataType)
        return Y_hat

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root.__print__()