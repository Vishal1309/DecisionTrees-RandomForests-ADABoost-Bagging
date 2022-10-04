from cmath import inf
from ctypes import byref
from distutils.log import info
from math import log, e
import numpy as np
import pandas as pd
np.random.seed(42)

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    # S = summation( (xi) * log(xi) / log(b) )
    value, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    S = 0
    base = 2
    for xi in probs:
        S -= xi * log(xi, base)
    return S
    pass

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    # G = 1 - (a / (a + b))^2 - (b / (a + b))^2;
    values, counts = np.unique(Y, return_counts = True)
    probs = counts / counts.sum()
    G = 1
    sq_sum = np.sum(np.square(probs))
    G -= sq_sum
    return G
    pass

def information_gain(Y, attr, criterion=None):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    # Discrete Input and Discrete output:
    if (Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "information_gain"):
        # Information_Gain = S(attr) - sum(|Tv| / T * S[value of attr])
        Entropy_Y = entropy(Y)
        class_type = np.unique(attr)
        Total_labels = Y.size
        to_subtract = 0
        for i in class_type:
            Y_curr_class = Y[attr == i]
            Entropy_curr_class = entropy(Y_curr_class)
            to_subtract += (Y_curr_class.size / Total_labels) * Entropy_curr_class # Tv / T * S(V)
        return Entropy_Y - to_subtract
    elif (Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "gini_index"):
        # Information_Gain = S(attr) - sum(|Tv| / T * S[value of attr])
        Entropy_Y = gini_index(Y)
        class_type = np.unique(attr)
        Total_labels = Y.size
        to_subtract = 0
        for i in class_type:
            Y_curr_class = Y[attr == i]
            Entropy_curr_class = gini_index(Y_curr_class)
            to_subtract += (Y_curr_class.size / Total_labels) * Entropy_curr_class # Tv / T * S(V)
        return Entropy_Y - to_subtract
    # Real Input and Discrete Output:
    elif (Y.dtype.name == "category" and attr.dtype.name == "float64" and criterion == "information_gain"):

        combined = pd.concat([attr, Y], axis = 1).reindex(attr.index)
        combined.columns = ['attr', 'label']
        sorted = combined.sort_values(by = ['attr', 'label'])
        best_split = 0
        best_gain = -np.inf
        attributes = sorted['attr'].to_numpy()
        Labels = sorted['label'].to_numpy()
        # print(attributes)
        # print(Labels)
        original_entropy = entropy(Labels)
        for i in range(1, len(attributes)):
            curr_split = float(attributes[i] + attributes[i-1]) / 2
            left = sorted[sorted['attr'] <= curr_split]['label'].to_numpy()
            right = sorted[sorted['attr'] > curr_split]['label'].to_numpy()
            G = original_entropy
            G -= (left.size / Labels.size) * (entropy(left))
            G -= (right.size / Labels.size) * (entropy(right))
            if(G > best_gain):
                best_gain = G
                best_split = curr_split
        return {best_gain, best_split}
    elif (Y.dtype.name == "category" and attr.dtype.name == "float64" and criterion == "gini_index"):
        combined = pd.concat([attr, Y], axis = 1).reindex(attr.index)
        combined.columns = ['attr', 'label']
        sorted = combined.sort_values(by = ['attr', 'label'])
        best_split = 0
        best_gain = -np.inf
        attributes = sorted['attr'].to_numpy()
        Labels = sorted['label'].to_numpy()
        # print(attributes)
        # print(Labels)
        original_entropy = gini_index(Labels)
        for i in range(1, len(attributes)):
            curr_split = float(attributes[i] + attributes[i-1]) / 2
            left = sorted[sorted['attr'] <= curr_split]['label'].to_numpy()
            right = sorted[sorted['attr'] > curr_split]['label'].to_numpy()
            G = original_entropy
            G -= (left.size / Labels.size) * (gini_index(left))
            G -= (right.size / Labels.size) * (gini_index(right))
            if(G > best_gain):
                best_gain = G
                best_split = curr_split
        return {best_gain, best_split}
    # Discrete Input and Real Output
    elif (Y.dtype.name == "float64" and attr.dtype.name == "category"):
        attribute_classes = np.unique(attr)
        G = np.var(Y)
        for class_type in attribute_classes:
            Y_curr_class = Y[attr == class_type]
            var_curr_class = np.var(Y_curr_class)
            weight = Y_curr_class.size / Y.size
            G -= weight * var_curr_class
        return G
    elif (Y.dtype.name == "float64" and attr.dtype.name == "float64"):
        combined = pd.concat([attr, Y], axis = 1).reindex(attr.index)
        combined.columns = ['attr', 'value']
        sorted = combined.sort_values(by = 'attr')
        best_split = 0
        best_gain = -np.inf
        overall_var = np.var(Y)
        attributes = sorted['attr'].to_numpy()
        values = sorted['value'].to_numpy()
        for i in range(1, len(attributes)):
            curr_split = float(attributes[i] + attributes[i-1]) / 2
            left = sorted[sorted['attr'] <= curr_split]['value'].to_numpy()
            right = sorted[sorted['attr'] > curr_split]['value'].to_numpy()
            G = overall_var
            G -= (left.size / values.size) * (np.var(left))
            G -= (right.size / values.size) * (np.var(right))
            if(G > best_gain):
                best_gain = G
                best_split = curr_split
        return {best_gain, best_split}
    pass

# N = 30
# P = 5
# Discrete Input and Discrete Output
# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randint(P, size = N),dtype="category")
# print(gini_index(y))

# Real Input and Discrete Output
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randint(P, size = N), dtype="category")
# print(information_gain(y, X[0]))

# Discrete Input and Real Output
# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randn(N))
# print(information_gain(y, X[0]))

# Real Input and Real Output
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randn(N))
# print(information_gain(y, X[0]))


