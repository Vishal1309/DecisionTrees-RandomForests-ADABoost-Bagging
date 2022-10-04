# METRICS

import math
import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    True_Positive = 0
    Total = y_hat.size
    for i in range(y.size):
        if y_hat[i] == y[i]:
            True_Positive += 1
    if Total == 0:
        return 1
    return float(True_Positive) / Total
    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    True_Positive = 0
    Total_Predicted_Positive = 0
    for i in range(y.size):
        if y_hat[i] == cls and y[i] == cls:
            True_Positive += 1
        if y_hat[i] == cls:
            Total_Predicted_Positive += 1
    if Total_Predicted_Positive == 0:
        return 1
    return float(True_Positive) / Total_Predicted_Positive
    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    Actual_Positives = 0
    Correctly_Predicted_Positive = 0
    for i in range(y.size):
        if y[i] == cls and y_hat[i] == cls:
            Correctly_Predicted_Positive += 1
        if y[i] == cls:
            Actual_Positives += 1
    if Actual_Positives == 0:
        return 1
    return float(Correctly_Predicted_Positive) / Actual_Positives
    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    sq = 0;
    for i in range(y.size):
        sq += (y[i] - y_hat[i]) ** 2
    mean_sq = sq / y.size
    return float(math.sqrt(float(mean_sq)))
    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    abs_sum = 0
    for i in range(y.size):
        abs_sum = abs(y[i] - y_hat[i])
    return float(abs_sum) / y.size
    pass
