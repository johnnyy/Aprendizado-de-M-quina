import numpy as np

def mse(y,y_pred):
    n=y.shape[0]
    return (np.sum((y-y_pred)**2))/n

def rmse(y,y_pred):
    return np.sqrt(mse(y,y_pred))
                
def mae(y,y_pred):
    n = y.shape[0]
    return np.sum(np.abs(y - y_pred))/n

def accuracy(x):
    return np.sum(np.diagonal(x))/np.sum(x)

def precision(x,ind):
    return x[ind,ind]/np.sum(x[ind,:])

def recall(x, ind):
    return x[ind,ind]/np.sum(x[:,ind])

def f1_measure(x, ind):
    r = recall(x, ind)
    p = precision(x, ind)
    return 2*r*p/(r + p)
    