import numpy as np
import scipy.stats as stats

# på mikaels klass: ändade så att fit tar emot X och y, och att predict tar emot X_new.
# då måste n och d vara properties som beräknas i fit, så att de kan användas i variance och stddev metoderna. 

# class
class LinearRegression1:
    def __init__(self): #constructor. self - refers to the object/instance 
        self.b = None # initialize coefficients to None, will be calculated in fit method
        self._X = None # store training data for later use in variance and stddev calculations
        self._y = None # store target variable for later use in variance and stddev calculations
        
    @property
    def n(self):
        return None if self._X is None else int(self._X.shape[0]) #self.d = None # number of features
    
    @property
    def d(self):
        return None if self._X is None else int(self._X.shape[1] -1)   #self.n = None # number of observations
  
    # OLS  estimation method - calculates coefficients, beta hat
    def fit(self, X, y): # think that this method fits a hyperplane to data points in a way that minimizes residuals
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1) # reshape y to column vector
     
        X1 = np.column_stack([np.ones(X.shape[0]), X]) # add column of ones for intercept term

        self._X = X1 
        self._y = y

        # beta_hat = (X^T * X)^-1 * X^T * y (matrix form) 
        XtX_inv =  np.linalg.inv(X1.T@X1) # (X^T * X)^-1
        
        self.beta = XtX_inv @ X1.T @ y # beta_hat = (X^T * X)^-1 * X^T * y  
                                                # self.beta is the vector of estimated coefficients
                                                # a vector of length d+1 (including intercept)
        y_hat = X1 @ self.beta # predicted values (matrix multiplication) 
        self.residuals = y - y_hat # squared vertical distances 
        # differences between actual and predicted values 
                                        # residuals will be used in variance and stddev calculations
        return self 
    
    def predict(self, X_new):
        # add intercept term to X_new
        n_new = X_new.shape[0]
        X_new = np.column_stack([np.ones(n_new), X_new]) # add column of ones for intercept
        return X_new @ self.beta # predicted values for new data points 
 
    def variance(self):
        # step 1: compute SSE (sum of squared errors)
        SSE = np.sum(self.residuals**2) # sum of squared residuals
        # step 2: divide by n-d-1 to get unbiased estimate of variance (degrees of freedom correction) 
        return SSE / (self.n - self.d - 1)
    
    # standard deviation method - this is just square root of variance, easy peasy!
    def standard_deviation(self):
       return np.sqrt(self.variance())
        
    # RMSE method - root mean squared error
    def root_mean_squared_error(self):
        SSE = np.sum(self.residuals**2)
        return np.sqrt(SSE / self.n)

    def significance_regression(self):
        #predicted values
        y_hat = self._X @ self.beta
        # mean of y
        np.mean(self._y) 
        # sums of squares 
        SSE = np.sum((self._y - y_hat)**2) # residuals
        SSR = np.sum((y_hat - np.mean(self._y))**2) # regression
        # degrees of freedom
        df_reg = self.d # number of predictors 
        df_err = self.n - self.d - 1
        # F-statistic
        F = (SSR / df_reg) / (SSE / df_err)
        # p-value
        p_value = 1 - stats.f.cdf(F, df_reg, df_err)

        return F, p_value 

    # method for relevence of the regression - R^2
    

    # significance test on individual variables - t-test on coefficients

    # method for calculating Pearson number between all pairs of parameters

    # confidence intervals on individual parameters

    # being able to set a confidence interval for the statistics??1