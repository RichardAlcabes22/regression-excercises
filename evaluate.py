import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

###########################################################################
def select_kbest(X_train,y_train,top_k=3):
    kbest = SelectKBest(f_regression,k=top_k)
    _ = kbest.fit(X_train,y_train)
    
    return X_train.columns[kbest.get_support()]

def rfe(X_train,y_train,top_k=3):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=top_k)
    rfe.fit(X_train,y_train)
    return X_train.columns[rfe.get_support()]

###########################################################################
def plot_residuals(y, yhat):
    '''
    a function to take in
      an array/Series of target "y" values, followed by
      an array/Series of predicted "yhat" values, and outputs
      a scatterplot/lineplot showing the residuals (errors) of the model
    '''
    plt.scatter(y,(yhat-y))
    plt.plot(y,yhat,c='red')
    plt.xlabel(f'x={y}')
    plt.ylabel(f'y=residual')
    plt.title('OLS_Linear Model')
    plt.show()


###########################################################################

def regression_errors(y,yhat):
    '''
    a function to take in
      an array/Series of target "y" values, followed by
      an array/Series of predicted "yhat" values, and outputs
      the SSE SST SSR MSE and RMSE of the model
    
    '''
    SSE = mean_squared_error(y,yhat) * len(y)
   
    SST = mean_squared_error(y,y.mean()) * len(y)
    SSR = SST - SSE
    MSE = mean_squared_error(y,yhat,squared=False)
    RMSE = mean_squared_error(y,yhat)

    return SSE,SST,SSR,MSE,RMSE
##########################################################################

def baseline_mean_errors(y):
    '''
    a function to take in
      an array/Series of target "y" values, and outputs
      the SSE MSE and RMSE of the BASELINE model    
    
    '''

    SSE = (y - y.mean()).sum()
    MSE = SSE / len(y)
    RMSE = MSE**0.5

    return SSE,MSE,RMSE

########################################################################

def better_than_baseline(y, yhat):
    '''
    a function to take in
      an array/Series of target "y" values, and compare it to
      the BASELINE model
    
    '''

    SSE = mean_squared_error(y,yhat) * len(y)
    SST = mean_squared_error(y,y.mean()) * len(y)
    SSR = SST - SSE 

    if SSR > 0:
        return "True, the model performs better than the baseline"
    else:
        return "False, the model displays no improvement upon baseline"
    

