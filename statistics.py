import numpy as np
import scipy.stats as stats


# class
class LinearRegression:
    def __init__(self, X, y): #constructor. self - refers to the object/instance 
        
        self.y = y #y is the target variable, ex median_house_value 
                   #it is a 1d column vector containing the values I want to predict
        self.n = X.shape[0] # number of observations
        self.d = X.shape[1] # number of features
        # adding intercept term (bias) to X - a column of ones as the first column of X
        self.X = np.column_stack([np.ones(self.n), X])
         # X is the design matrix of predictors (features). 
                   # 2d, columns are features, rows observations

    # OLS  estimation method
    def Ordinary_least_squares(): 
        

# predicted values and residuals 


# sample variance method 
    def sample_variance():

# standard deviation method 
    def standard_deviation():

# RMSE method 
    def RMSE():