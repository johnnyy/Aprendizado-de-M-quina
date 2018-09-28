import numpy as np

class SimpleLinearRegression(object):
        
    def fit(self,X_,y):
        X = X_[:,0]
        self.b1_ = np.sum(((X- np.mean(X))*(y-np.mean(y))))/ np.sum((X- np.mean(X))**2)
        self.b0_ = np.mean(y) - self.b1_*np.mean(X)
        
    def predict(self,X):
        return self.b1_*X[:,0] + self.b0_

    
    
class LogisticRegression(object):
    def fit(self,X,y,learning_rate,epochs):
        num_pos = X.shape[0]
        x = np.hstack((np.ones(num_pos).reshape(num_pos, 1), X)) # add 1 for beta_0 intercept
        y_ = y.reshape(num_pos, 1) # reshape y to make 2D shape (n, 1)
        self.beta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)

        for step in np.arange(epochs):
            x_beta = np.dot(x, self.beta)
            y_hat = 1 / (1 + np.exp(-x_beta))
            preds = np.round( y_hat )
            gradient = np.dot(np.transpose(x), y_ - y_hat)
            self.beta = self.beta + learning_rate*gradient
        print("Sucess")
    def predict(self,X):
        num_pos = X.shape[0]
        x = np.hstack((np.ones(num_pos).reshape(num_pos, 1), X))
        x_beta = np.dot(x, self.beta)
        y_hat = 1 / (1 + np.exp(-x_beta))
        return np.round( y_hat ).reshape(num_pos,)