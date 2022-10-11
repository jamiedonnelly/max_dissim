import numpy as np 
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import imageio
import os

def max_dissim(A: np.array, B: np.array, n: int):
    """
    Summary:
        Algorithm used to iteratively add datapoints from a dataset B to a subset A based on a maximally dissimilar criterion. Given
        an integer, n, denoting number of additional points to add, datapoints in B are iteratively added to A based on maximising the
        euclidean distance from points already in A. 

    Args:
        A (np.array): Starting subset from which the first maximally dissimilar point in remaining dataset in B is determined by. 
        B (np.array): Remaining (if A begins as a subset of B) dataset or separate dataset from which additional points are selected
                      from and added to dataset A.
        n (int): Additional number of points to select from B to be added to A.
    """

    new_a = A  
    new_b = B

    for i in range(n):
        dist = distance_matrix(new_a,new_b)
        di_sub = np.min(dist,axis=0)
        z = np.argmax(di_sub)
        new_a = np.concatenate([new_a,new_b[z].reshape(1,-1)],axis=0)
        new_b = np.delete(new_b,z,axis=0)
    
    return new_a


if __name__=="__main__":

    a = np.random.randn(1,2)
    b = np.random.randn(50,2)

    n = 10

    new_a = a 
    new_b = b

    for i in range(1,n+1):

        dist = distance_matrix(new_a,new_b)
        di_sub = np.min(dist,axis=0)
        z = np.argmax(di_sub)

        plt.figure()
        plt.scatter(new_a[:,0],new_a[:,1])
        plt.scatter(np.delete(new_b[:,0],z),np.delete(new_b[:,1],z),alpha=0.7)
        plt.scatter(new_b[z,0],new_b[z,1],marker='x',color='red')
        plt.legend(['A','B','Z'])
        plt.title(f"Iteration {i}")
        plt.savefig(f'/.../MDA/frame{i}.png')

        new_a = np.concatenate([new_a,new_b[z].reshape(1,-1)],axis=0)
        new_b = np.delete(new_b,z,axis=0)


    ims = [imageio.imread(os.path.join('/.../MDA/',f)) for f in sorted(os.listdir('/.../MDA/'))]
    imageio.mimwrite('/.../MDA.gif', ims, format='GIF', fps='1')