import numpy as np

def mse(y,y_pred):
    n=y.shape[0]
    return (np.sum((y-y_pred)**2))/n

def rmse(y,y_pred):
    return np.sqrt(mse(y,y_pred))
                
def mae(y,y_pred):
    n = y.shape[0]
    return np.sum(np.abs(y - y_pred))/n
    