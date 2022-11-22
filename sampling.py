import numpy as np 
import plotly
from scipy.spatial import distance_matrix
import matplotlib.animation as animation

class MDASampler():
    
    def __init__(self,initial_data,pool_data,initial_index=False):
        self.data, self.initial_data = initial_data, initial_data
        self.pool, self.initial_pool = pool_data, pool_data
        if initial_index:
            self.ix = np.array(initial_index,dtype=int)
        else:
            self.ix = np.array([],dtype=int)
        
    def _findix(self,index):
        ix_eq = np.argwhere(self.initial_pool==self.pool[index])
        self.ix = np.concatenate([self.ix,np.intersect1d(ix_eq[0],ix_eq[1])],dtype=int)

    def _sample(self,n):
        while self.ix.shape[0]<n:
            distmat = distance_matrix(self.data,self.pool)
            min_vec = np.min(distmat,axis=0)
            max_ix = np.argmax(min_vec)
            print(max_ix)
            self._findix(max_ix) # Append to indices 
            self.data = np.concatenate([self.data,self.pool[max_ix].reshape(1,-1)],axis=0) # Update data
            self.pool = np.delete(self.pool,max_ix,axis=0) # Update pool
    
    def __call__(self,n):
        self._sample(n)


class MDA(MDASampler):
    
    def __init__(self, X, y, n, random_init=False, init_size=100):
        
        self.X, self.y = X, y
         
        if random_init:
            self.initial_data, self.initial_pool = MDA._initialise(X,y,init_size)
        else:
            self.initial_data, self.initial_pool = MDA._initialise(X,y)
        
        super(MDA,self).__init__(self.initial_data,self.initial_pool)
    
    @staticmethod
    def _initialise(X,y,n=1):
        init_ix = np.array(np.random.choice(X.shape[0]))
        init_data = X[init_ix]
        init_pool = np.delete(y,init_ix,axis=0)
        return init_data.reshape(n,-1), init_pool
        
    def __call__(self,n):
        super().__call__(n)
        return self.X[self.ix], self.y[self.ix]
    

def lhs(data,N):
    """
    Summary:
        Algorithm to calculate a latin-hypercube data sample from a set of empirical data with unknown distribution. 
        Since explicit marginal distributions of features are not known, empirical CDFs are calculated and each marginal is sampled from. 
        If there are 5 features and a sample of N=100 is obtained, floor(N^(1/5)) data points will be sampled from each marginal. 

    Args:
        data (_type_): _description_
        N (_type_): _description_
    """
    pass