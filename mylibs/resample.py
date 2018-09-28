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

def split_stratified_train_test(y, perc_train, seed):
    
    unique, count = np.unique(y, return_counts=True)
    train_A =  int(perc_train*count[0])
    train_B =  int(perc_train*count[1])
    count_A = 0
    count_B = 0
    idx_train = []
    idx_test = []
    for i in range(y.shape[0]):
        if (y[i] == unique[0]):
            if(count_A <= train_A):
                idx_train.append(i)
                count_A +=1
            else:
                idx_test.append(i)
        elif(y[i] == unique[1]):
            if(count_B <= train_B):
                idx_train.append(i)
                count_B +=1
            else:
                idx_test.append(i)


    if(seed):
        np.random.seed(seed)
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)
    return idx_train,idx_test


    
        
    
    