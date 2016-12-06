import numpy as np
import math

def Random_Projection(M, new_dim, prng):
    ''' Project *columns* of input matrix M into lower-dimension new_dim

       Args
       --------
       M : 2D array, old_dim x nFeatures

       Returns
       --------
       M_red : 2D array, new_dim x nFeatures

       Internals
       --------
       Creates R : 2D array, new_dim x old_dim
                   projection matrix
                   each column is a random vector of size new_dim,
                      with entries in {-sqrt(3), 0, sqrt(3)}
                           with associated probabilities 1/6, 2/3, 1/6
    '''
    old_dim = M[:, 0].size
    p = np.array([1./6, 2./3, 1./6])
    c = np.cumsum(p)
    randdoubles = prng.random_sample(new_dim*old_dim)
    R = np.searchsorted(c, randdoubles)
    R = math.sqrt(3)*(R - 1)
    R = np.reshape(R, (new_dim, old_dim))
    
    M_red = np.dot(R, M)
    return M_red
