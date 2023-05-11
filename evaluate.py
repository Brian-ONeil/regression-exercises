#####imports

import numpy as np
import os
import seaborn as sns
import scipy.stats as stat
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import warnings
warnings.filterwarnings("ignore")
import wrangle as wra
import env
import explore as exp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif

####functions

def plot_residuals(y, yhat):
    '''creates a residual plot'''
    
    # scatter is my actuals
    plt.scatter(train.squarefeet, train.residuals)

    # lineplot is my regression line
    plt.plot(train.squarefeet, train.yhat)

    plt.xlabel('x = squarefeet')
    plt.ylabel('y = residuals')
    plt.title('OLS linear model (n = 5000)')
    #plt.text(5000, 5000000, 'n = 5000', fontsize=12, color='red')

    plt.show()
    
def regression_errors(y, yhat):
    """
    Calculate regression error metrics for a given set of actual and predicted values.
    
    Parameters:
    y (array-like): Actual values of the response variable
    yhat (array-like): Predicted values of the response variable
    
    Returns:
    tuple: A tuple of the sum of squared errors (SSE), explained sum of squares (ESS),
           total sum of squares (TSS), mean squared error (MSE), and root mean squared error (RMSE)
    """
    # Calculate SSE, ESS, TSS, MSE, and RMSE
    SSE = np.sum((y - yhat)**2)
    ESS = np.sum((yhat - np.mean(y))**2)
    TSS = SSE + ESS
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    
    # Return the results as a tuple
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    """
    Calculate regression error metrics for the baseline model that always predicts the mean of y.
    
    Parameters:
    y (array-like): Actual values of the response variable
    
    Returns:
    tuple: A tuple of the sum of squared errors (SSE), mean squared error (MSE), and root mean squared error (RMSE)
    """
    # Calculate the mean of y
    mean_y = np.mean(y)
    
    # Calculate SSE, MSE, and RMSE for the baseline model
    SSE = np.sum((y - mean_y)**2)
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    
    # Return the results as a tuple
    return SSE, MSE, RMSE

def better_than_baseline(y, yhat):
    """
    Check if the sum of squared errors (SSE) for the model is less than the SSE for the baseline model.
    
    Parameters:
    y (array-like): Actual values of the response variable
    yhat (array-like): Predicted values of the response variable
    
    Returns:
    bool: True if the SSE for the model is less than the SSE for the baseline model, otherwise False
    """
    # Calculate SSE for the model and the baseline model
    SSE_model = np.sum((y - yhat)**2)
    mean_y = np.mean(y)
    SSE_baseline = np.sum((y - mean_y)**2)
    
    # Check if the SSE for the model is less than the SSE for the baseline model
    if SSE_model < SSE_baseline:
        return True
    else:
        return False

def select_kbest(X, y, k):
    """
    Select the top k features based on the SelectKBest class and return their names.
    
    Parameters:
    X (array-like): The predictors
    y (array-like): The target variable
    k (int): The number of features to select
    
    Returns:
    list: A list of the names of the top k selected features
    """
    # Create a SelectKBest object and fit it to the data
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    
    # Get the indices of the top k selected features
    idxs_selected = selector.get_support(indices=True)
    
    # Get the names of the top k selected features
    features_selected = list(X.columns[idxs_selected])
    
    # Return the names of the top k selected features
    return features_selected

def rfe(X, y, k):
    """
    Select the top k features based on the RFE class and return their names.
    
    Parameters:
    X (array-like): The predictors
    y (array-like): The target variable
    k (int): The number of features to select
    
    Returns:
    list: A list of the names of the top k selected features
    """
    # Create a linear regression model
    model = LinearRegression()
    
    # Create an RFE object and fit it to the data
    selector = RFE(model, n_features_to_select=k)
    selector.fit(X, y)
    
    # Get the indices of the top k selected features
    idxs_selected = selector.get_support(indices=True)
    
    # Get the names of the top k selected features
    features_selected = list(X.columns[idxs_selected])
    
    # Return the names of the top k selected features
    return features_selected

