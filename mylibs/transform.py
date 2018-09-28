import numpy as np


def standardize(X):
    X_stand = np.copy(X)
    n_cols = X.shape[1]
    for i in range(n_cols):
        desvio = np.std(X[:,i])
        media = np.mean(X[:,i])
        X_stand[:,i] = (X[:,i] - media)/desvio
    return X_stand

def normalize(X):
    X_norm = np.copy(X)
    n_cols = X.shape[1]

    for i in range(n_cols):
        minimo = np.min(X[:,i])
        maximo = np.max(X[:,i])
        X_norm[:,i] = (X[:, i] - minimo) / (maximo - minimo)
    
    return X_norm