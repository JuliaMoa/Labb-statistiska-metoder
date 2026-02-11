import numpy as np
import scipy.stats as stats

class LinearRegression1:
    def __init__(self, alpha=0.05): 
        self.b = None 
        self._X = None 
        self._y = None 
        self.SSE = None 
        self.Syy = None
        self.XtX_inv = None
        self.sigma2_hat = None 
        self.alpha = alpha

    @property
    def n(self):
        return None if self._X is None else int(self._X.shape[0]) 
    
    @property
    def d(self):
        return None if self._X is None else int(self._X.shape[1] -1)  
    
    def fit(self, X, y): 
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1) # ensure y is a column vector
     
        X1 = np.column_stack([np.ones(X.shape[0]), X]) 

        self._X = X1 
        self._y = y

        XtX_inv =  np.linalg.inv(X1.T@X1)
        self.XtX_inv = XtX_inv
        self.beta = XtX_inv @ X1.T @ y # beta_hat = (X^T * X)^-1 * X^T * y  
                                                # self.beta is the vector of estimated coefficients
                                                
        y_hat = X1 @ self.beta # predicted values (matrix multiplication) 
        self.residuals = y - y_hat # squared vertical distances - differences between actual and predicted values 
                                    
        SSE = np.sum(self.residuals**2) # sum of squared residuals
        self.SSE = SSE
        y_mean = np.mean(self._y)
        Syy = np.sum((self._y - y_mean)**2)
        self.Syy = Syy  # total variation 
        sigma2_hat = self.variance()
        self.sigma2_hat = sigma2_hat 
        return self 
    
    def predict(self, X_new):
        n_new = X_new.shape[0]
        X_new = np.column_stack([np.ones(n_new), X_new]) # add column of ones for intercept
        return X_new @ self.beta # predicted values for new data points 
 
    def variance(self):
        return self.SSE / (self.n - self.d - 1)
    
    def standard_deviation(self):
        return np.sqrt(self.variance())
        
    def root_mean_squared_error(self):
        return np.sqrt(self.SSE / self.n)
    
    def significance_regression(self):
        SSR = self.Syy - self.SSE # regression
        # degrees of freedom
        df_reg = self.d 
        df_err = self.n - self.d - 1
        # F-statistic
        F = (SSR / df_reg) / (self.SSE / df_err)
        # p-value
        p_value = stats.f.sf(F, df_reg, df_err) # survival function for F-distribution
        #significant = p_value < self.alpha
        return F, p_value

    def r_squares(self):  # method for relevence of the regression 
        return 1 - self.SSE / self.Syy # R^2 = 1 - SSE/Syy

    def t_test_coefficiants(self): 
        t_stats = self.beta.flatten() / np.sqrt(self.sigma2_hat * np.diag(self.XtX_inv)) # t-statistics for each coefficient
        p_values = 2 * stats.t.sf(np.abs(t_stats), df= self.n - self.d - 1) # two-tailed p-values for t-test 
        return t_stats, p_values
    
    def Pearson_correlation(self):
        X_features = self._X[:, 1:] # exclude intercept term
        corr_matrix = np.corrcoef(X_features, rowvar=False)
        return corr_matrix
    
    def confidence_intervals(self):
        SE = np.sqrt(np.diag(self.XtX_inv) * self.sigma2_hat) # vector of standard errors of all coefficients
        df_residual = self.n - self.d - 1
        t_critical = stats.t.ppf(1 - self.alpha/2, df=df_residual) 

        lower = self.beta.flatten() - t_critical * SE 
        upper= self.beta.flatten() + t_critical * SE
        return lower, upper
   