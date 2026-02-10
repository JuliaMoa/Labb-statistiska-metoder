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
        self.SSE = None 
        self.Syy = None
        self.XtX_inv = None
        # Dessa ses av alla metoder! 

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
        self.XtX_inv = XtX_inv
        
        self.beta = XtX_inv @ X1.T @ y # beta_hat = (X^T * X)^-1 * X^T * y  
                                                # self.beta is the vector of estimated coefficients
                                                # a vector of length d+1 (including intercept)
        y_hat = X1 @ self.beta # predicted values (matrix multiplication) 
        self.residuals = y - y_hat # squared vertical distances 
        # differences between actual and predicted values 
                                        # residuals will be used in variance and stddev calculations

        # SSE and Syy - handy for later
        SSE = np.sum(self.residuals**2) # sum of squared residuals
        self.SSE = SSE
        y_mean = np.mean(self._y)
        Syy = np.sum((self._y - y_mean)**2)
        self.Syy = Syy  # total variation 

        return self 
    
    def predict(self, X_new):
        # add intercept term to X_new
        n_new = X_new.shape[0]
        X_new = np.column_stack([np.ones(n_new), X_new]) # add column of ones for intercept
        return X_new @ self.beta # predicted values for new data points 
 
    def variance(self):
        # step 1: compute SSE (sum of squared errors) - already done in fit 
        # step 2: divide by n-d-1 to get unbiased estimate of variance (degrees of freedom correction) 
        return self.SSE / (self.n - self.d - 1)
    
    # standard deviation method - this is just square root of variance, easy peasy!
    def standard_deviation(self):
        return np.sqrt(self.variance())
        
    # RMSE method - root mean squared error
    def root_mean_squared_error(self):
        return np.sqrt(self.SSE / self.n)
    
    # ---------------- VG ------------------------------------------------------------------------------

    def significance_regression(self):
        #predicted values
        y_hat = self._X @ self.beta
        SSR = self.Syy - self.SSE # regression
        # degrees of freedom
        df_reg = self.d # number of predictors 
        df_err = self.n - self.d - 1
        # F-statistic
        F = (SSR / df_reg) / (self.SSE / df_err)
        # p-value
        p_value = stats.f.sf(F, df_reg, df_err) # survival function for F-distribution
        return F, p_value 

    # method for relevence of the regression 

    def r_squares(self):
        y_mean = np.mean(self._y)
        return 1 - self.SSE / self.Syy # R^2 = 1 - SSE/Syy

    # significance test on individual variables - t-test on coefficients
    def t_test_coefficiants(self): 
        sigma2_hat = self.variance() # estimated variance of residuals
        t_stats = self.beta.flatten() / np.sqrt(sigma2_hat * np.diag(self.XtX_inv)) # t-statistics for each coefficient
        p_values = 2 * stats.t.sf(np.abs(t_stats), df= self.n - self.d - 1) # two-tailed p-values for t-test
        return t_stats, p_values # t-stats: t-values for each coefficient, p-values: significance of each coefficient

    # method for calculating Pearson number between all pairs of parameters

    

    # confidence intervals on individual parameters

    # being able to set a confidence interval for the statistics??