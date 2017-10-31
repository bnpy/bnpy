'''
Normal1DK3

Simple toy dataset of 3 comps from
  1D normal distributions with different means
'''

import numpy as np

from bnpy.data import XData

# User-facing fcns
###########################################################


def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    '''
      Args
      -------
      seed : integer seed for random number generator,
              used for actually *generating* the data
      nObsTotal : total number of observations for the dataset.

      Returns
      -------
        Data : bnpy XData object, with nObsTotal observations
    '''
    X, TrueZ = generate_data(seed, nObsTotal)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_data_info():
    return 'Normal Data. %d true clusters.' % (3)

# Generate Raw Data
###########################################################
mu = np.asarray([-10, 0, 10])


def generate_data(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    K = mu.size
    N = nObsTotal / K
    Xlist = list()
    Zlist = list()
    for k in range(K):
        if k == K - 1:
            N = nObsTotal - (K - 1) * N
        X = mu[k] + PRNG.randn(N, 1)
        Xlist.append(X)
        Zlist.append(k * np.ones(N))
    X = np.vstack(Xlist)
    TrueZ = np.hstack(Zlist)

    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ
