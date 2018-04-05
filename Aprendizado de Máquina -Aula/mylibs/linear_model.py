import numpy as np

class SimpleLinearRegression(object):
        
    def fit(self,X_,y):
        X = X_[:,0]
        self.b1_ = np.sum(((X- np.mean(X))*(y-np.mean(y))))/ np.sum((X- np.mean(X))**2)
        self.b0_ = np.mean(y) - self.b1_*np.mean(X)
        
    def predict(self,X):
        return self.b1_*X[:,0] + self.b0_