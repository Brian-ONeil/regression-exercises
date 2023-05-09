#######IMPORTS

import numpy as np
import os
import seaborn as sns
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import warnings
warnings.filterwarnings("ignore")
import wrangle as wra
import env
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

#######FUNCTIONS

def plot_variable_pairs(df):
    sns.pairplot(df.sample(100000), corner=True, kind='reg')
    plt.show()
        
def plot_categorical_and_continuous_vars(df, cont_var, cat_var):
    # Plot a boxplot of the continuous variable for each category
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=cat_var, y=cont_var, data=df)
    plt.title(f'{cont_var} by {cat_var}')
    plt.show()
    
    # Plot a violinplot of the continuous variable for each category
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=cat_var, y=cont_var, data=df)
    plt.title(f'{cont_var} by {cat_var}')
    plt.show()

    # Plot a swarmplot of the continuous variable for each category
    plt.figure(figsize=(8, 6))
    sns.swarmplot(x=cat_var, y=cont_var, data=df)
    plt.title(f'{cont_var} by {cat_var}')
    plt.show()
    
