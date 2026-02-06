import numpy as np
import scipy.stats as stats

# class
class LinearRegression1:
    def __init__(self, X, y): #constructor. self - refers to the object/instance 
        
        self.y = y #y is the target variable, ex median_house_value 
                   #it is a 1d column vector containing the values I want to predict
        self.n = X.shape[0] # number of observations
        self.d = X.shape[1] # number of features
        # adding intercept term (bias) to X - a column of ones as the first column of X
        self.X = np.column_stack([np.ones(self.n), X])
         # X is the design matrix of predictors (features). 
                   # 2d, columns are features, rows 
    
    # OLS  estimation method - calculates coefficients, beta hat
    def Ordinary_least_squares(self): # think that this method fits a hyperplane to data points in a way that minimizes residuals
        # beta_hat = (X^T * X)^-1 * X^T * y (matrix form) 
        XtX_inv =  np.linalg.inv(self.X.T@self.X) # (X^T * X)^-1
        self.beta = XtX_inv @ self.X.T @ self.y # beta_hat = (X^T * X)^-1 * X^T * y  
                                                # self.beta is the vector of estimated coefficients
                                                # a vector of length d+1 (including intercept)
        y_hat = self.X @ self.beta # predicted values (matrix multiplication) 
        self.residuals = self.y - y_hat # squared vertical distances 
       # differences between actual and predicted values 
                                        # residuals will be used in variance and stddev calculations
        return self 
 
    def variance(self):
        # step 1: compute SSE (sum of squared errors)
        SSE = np.sum(self.residuals**2) # sum of squared residuals
        # step 2: divide by n-d-1 to get unbiased estimate of variance (degrees of freedom correction) 
        return SSE / (self.n - self.d - 1)
    
    # standard deviation method - this is just square root of variance, easy peasy!
    def standard_deviation(self):
        np.sqrt(self.variance())

    # RMSE method - root mean squared error
    def root_mean_squared_error(self):
        SSE = np.sum(self.residuals**2)
        return np.sqrt(SSE / self.n)