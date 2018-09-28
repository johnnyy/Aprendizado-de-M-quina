import numpy as np
from mylibs import dist_metrics

class KNNClassifier(object):
    def __init__(self,k=5,metric='minkowski', p=2):
        self.k_ = k
        self.metric_ = metric
        self.p_ = p
            
        print("Sucess")
        
    def fit(self,X,y):
        self.X = X
        self.y = y
    def predict(self, value):
        prediction = []
        idx_sort = 0
        id_s=0
        for i in range(value.shape[0]):
            distances = 0
            if(self.metric_ == 'minkowski'):
                distances = dist_metrics.minkowski_distance(self.X, value[i], self.p_)
            elif(self.metric_ == 'euclidean' ):
                distances = dist_metrics.euclidean_distance(self.X,value[i])
            elif(self.metric_ == 'manhattan' ):
                distances = dist_metrics.manhattan_distance(self.X,value[i])
            elif(self.metric_ == 'chebyshev' ):
                distances = dist_metrics.chebyshev_distance(self.X,value[i])
            else:
                 print("ERRO metrics")
                 exit(-1)
            
            idx_s = np.argsort(distances)
            idx_sort = idx_s[1:self.k_+1]
            output_values = self.y[idx_sort]
            counts = np.unique(output_values, return_counts=True)
            idx_max = np.argmax(counts[1])
            prediction.append(counts[0][idx_max])
            #print("Distance:{}".format(distances))
            #print("Menor:{},Idex:{}. 2Menor{},Index{}".format(distances[idx_sort[0]],idx_sort[0],distances[idx_sort[1]],idx_sort[1]))
            #print('idx_sort:{}, output_values:{}, prediction:{}'.format(idx_sort, output_values, prediction[i]))
        return prediction
       
        
        
   
class KNNRegressor(object):
    
    def __init__(self,k=5,metric='minkowski', p=2):
        self.k_ = k
        self.metric_ = metric
        self.p_ = p
        print("Sucess")
        
    def fit(self,X,y):
        self.X = X
        self.y = y
    def predict(self, value):
        prediction = []
       
        for i in range(value.shape[0]):
            distances = 0
            if(self.metric_ == 'minkowski'):
                distances = dist_metrics.minkowski_distance(self.X, value[i], self.p_)
            elif(self.metric_ == 'euclidean' ):
                distances = dist_metrics.euclidean_distance(self.X,value[i])
            elif(self.metric_ == 'manhattan' ):
                distances = dist_metrics.manhattan_distance(self.X,value[i])
            elif(self.metric_ == 'chebyshev' ):
                distances = dist_metrics.chebyshev_distance(self.X,value[i])
            else:
                 print("ERRO Metrics")
                 exit(-1)
            
            idx_s = np.argsort(distances)
            idx_sort = idx_s[1:self.k_+1]
            output_values = self.y[idx_sort]
            prediction.append(np.sum(output_values) / output_values.shape[0])
            #print('idx_sort:{}, output_values:{}, prediction:{}'.format(idx_sort, output_values, prediction[i]))
        return prediction