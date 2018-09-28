import numpy as np

def minkowski_distance(X, row, p):
    val =[]
    for i in range(X.shape[0]):
        X_ = (np.sum(np.abs(X[i:] - row)**p))**(1.0/p)
        val.append(X_)
    return val
    
def euclidean_distance(X, row):
    return minkowski_distance(X,row,2)
def manhattan_distance(X, row):
    return minkowski_distance(X,row,1)

def chebyshev_distance(X, row):
    val = []
    for i in range(X.shape[0]):
        X_ = np.max(np.abs((X[i,:]-row)))
        val.append(X_)
    return val