import numpy as np

def mean(x):
    return np.sum(x)/x.shape[0]

def stdev(x):
    media = mean(x)
    return np.sqrt((1/x.shape[0])*np.sum((x-media)**2))

def var(x):
    return stdev(x)**2