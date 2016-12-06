'''
ToyRegression1D

Synthetic problem where both X and Y are 1D variables

We'll walk down the real line assigning a unit interval to every cluster
In that interval, we'll generate data from the 
appropriate linear section of a sawtooth wave
'''
import numpy as np
import bnpy
from matplotlib import pylab;

DEFAULT_SEED = 8675309
DEFAULT_N = 10000

def get_data(seed=DEFAULT_SEED, N=DEFAULT_N, **kwargs):
    ''' Generate toy data for regression problem

    Args
    -------
    seed : integer seed for random number generator,
          used for actually *generating* the data

    Returns
    -------
    Data : bnpy XData object
        includes X and Y attributes
    '''
    X, Y, TrueZ = make_toy_XY(N=DEFAULT_N, seed=DEFAULT_SEED)
    Data = bnpy.data.XData(X=X, Y=Y, TrueZ=TrueZ)
    Data.name = 'ToyRegression1D'
    Data.summary = ''
    return Data

def make_toy_XY(N=10000, K=6, w_mag=1, sigma=0.05, seed=DEFAULT_SEED):
    '''

    Returns
    -------
    X : 2D array
    Y : 1D array
    '''
    PRNG = np.random.RandomState(seed)
    assert K % 2 == 0

    Npercomp = N // K
    X = np.zeros((N,1))
    Y = np.zeros((N,1))
    TrueZ = np.zeros(N)

    b_val_Ko2 = np.arange(1, K//2+1)
    b_k_K = np.hstack([-1 * b_val_Ko2[::-1], b_val_Ko2])
    start = 0
    for k in range(K):
        stop = start + Npercomp

        x_min = -1 * (K/2 - k)
        b_k = b_k_K[k]
        if k % 2 == 0:
            w_k = w_mag
        else:
            w_k = - w_mag

        X[start:stop] = x_min + PRNG.rand(Npercomp, 1)
        Y[start:stop] = w_k * X[start:stop] + b_k
        Y[start:stop] += sigma * PRNG.randn(Npercomp, 1)
        TrueZ[start:stop] = k
        start += Npercomp
    return X, Y, TrueZ

if __name__ == '__main__':
    X, Y, TrueZ = make_toy_XY()
    pylab.plot(X, Y, 'k.')
    pylab.show()