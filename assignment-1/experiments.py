
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from metrics import *
from datetime import datetime

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

types = ["dido", "diro", "rido", "riro"] # type of experiments


def fake_data_gen(N=30, M=5, x_y_type="dido"):
    '''
    x_y_type = type of data in input and output
    '''
    N, M = int(N), int(M)
    P = 5  # No of categories
    # code from usage.py
    if x_y_type == "diro":
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    elif x_y_type == "dido":
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif x_y_type == "rido":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    else:
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))

    return X, y

def experimenting(N=[30, ], M=[5, ], Tree_Type="sklearn", x_y_types=["dido", ]):
    '''
    experimenting over different x_y_types and over sktree and mytree over differnt N and M values
    '''
    assert(len(N) == 1 or len(M) == 1)
    res = np.zeros((len(x_y_types), len(N), len(M), 4)) # storing results of all potential combinations
    for x_y_type in x_y_types:
        for idn, n in enumerate(N):
            for idm, m in enumerate(M):
                X, y = fake_data_gen(N=n, M=m, x_y_type=x_y_type) # generating fake data to feed into our trees

                if Tree_Type== "sklearn":
                    if x_y_type == "riro" or x_y_type == "diro": # need regressor for these types
                        tree = DecisionTreeRegressor()
                    else:
                        tree = DecisionTreeClassifier(criterion="entropy") # need classifier for this type
                else:
                    Tree_Type= "MyTree"
                    tree = DecisionTree() # my tree is supposed to be multifunctional, letsee
                st = datetime.now()
                tree.fit(X, y)
                en = datetime.now()
                learning = (en-st).total_seconds()

                st = datetime.now()
                tree.predict(X)
                en = datetime.now()
                predicting = (en-st).total_seconds()

                res[x_y_types.index(x_y_type), idn, idm] = np.array([n, m, learning, predicting]) # memoizing for later use while plotting

    # plt
    plt.figure()
    if len(N) > 1 or len(M) > 1:
        if len(N) <= 1:
            for x_y_type in x_y_types:
                plt.plot(res[x_y_types.index(x_y_type), 0, :, 1], res[x_y_types.index(x_y_type), 0, :, 2], label=x_y_type)
            plt.title("Plot for learning while varying M vs time for {}".format(Tree_Type))
            plt.xlabel("M")
            plt.ylabel("Time in seconds")
        else:
            for x_y_type in x_y_types:
                plt.plot(res[x_y_types.index(x_y_type), :, 0, 0], res[x_y_types.index(x_y_type), :, 0, 2], label=x_y_type)
            plt.title("Plot for learning while varying N vs time for {}".format(Tree_Type))
            plt.xlabel("N") 
            plt.ylabel("Time in seconds")
    plt.legend()
    plt.savefig("Learning_TIMES.png")

    plt.figure()
    if len(N) > 1 or len(M) > 1:
        if len(N) <= 1:
            for x_y_type in x_y_types:
                plt.plot(res[x_y_types.index(x_y_type), 0, :, 1], res[x_y_types.index(x_y_type), 0, :, 3], label=x_y_type)
            plt.title("Plot for lpredicting while varying M vs time for {}".format(Tree_Type))
            plt.xlabel("M")
            plt.ylabel("Time in seconds")
        else:
            for x_y_type in x_y_types:
                plt.plot(res[x_y_types.index(x_y_type), :, 0, 0], res[x_y_types.index(x_y_type), :, 0, 3], label=x_y_type)
            plt.title("Plot for predicting while varying N vs time for {}".format(Tree_Type))
            plt.xlabel("N")
            plt.ylabel("Time in seconds")
    plt.legend()
    plt.savefig("Predicting_TIMES.png")
    return res

experimenting(N=list(range(30, 81, 5)), x_y_types=types, Tree_Type="sklearn")
# experimenting(N=list(range(3, 20, 4), x_y_types=types, Tree_Type="sklearn")
# same way we can experiment using Tree_Type as "mine" after some debugging on mytree...
# experimenting(N=list(range(30, 81, 5)), x_y_types=types, Tree_Type="mine") # some error in tree at some point.. ( maybe while discrete input) not sure.. needs debugging