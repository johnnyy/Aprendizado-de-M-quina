import numpy as np
def split_train_test(n_elem, perc_train, seed):
    a = [ i for i in range(n_elem)]
    np.random.seed(seed)
    np.random.shuffle(a)
    elem1 = int(n_elem*perc_train)
    
    return a[:elem1],a[elem1:]

def split_k_fold(n_elem, n_splits, shuffle, seed):
    if(n_splits > 1):
        a = [ i for i in range(n_elem)]
        if(shuffle):
            
            np.random.seed(seed)
            np.random.shuffle(a)
        X_test = []
        X_train = []
        init_split = 0
        end_split = (n_elem/n_splits) -1
        for i in range(n_splits):
            test = []
            train = []
            for j in range(n_elem):
                if(j >= init_split and j <= end_split):
                    test.append(a[j])
                else:
                    train.append(a[j])
            X_test.append(test)
            X_train.append(train)
            init_split = init_split + (n_elem/n_splits)
            end_split = end_split + (n_elem/n_splits) 
        return X_train, X_test
       
        
    
    