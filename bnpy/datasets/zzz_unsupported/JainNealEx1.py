'''
JainNealEx1.py

Toy binary data from K=5 states.

Usage
---------
To take a look at the states, just run this script as an executable.
$ python JainNealEx1.py
'''
import numpy as np
import scipy.linalg

from bnpy.data import XData


def get_short_name():
    ''' Return short string used in filepaths to store solutions
    '''
    return 'JainNealEx1'


def get_data_info():
    return 'Toy Binary Data. K=%d true clusters. D=%d.' % (K, D)


def get_data(seed=8675309, nObsTotal=None, nPerState=20, **kwargs):
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
    if nObsTotal is not None:
        nPerState = nObsTotal // K
    X, TrueZ = genToyData(seed=seed, nPerState=nPerState)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


# Make A in 3D
#######################################################
K = 5
D = 6
phi = np.asarray([
    [.95, .95, .95, .95, .95, .95],
    [.05, .05, .05, .05, .95, .95],
    [.95, .05, .05, .95, .95, .95],
    [.05, .05, .05, .05, .05, .05],
    [.95, .95, .95, .95, .05, .05],
])


def genToyData(seed=1234, nPerState=20):
    '''
    '''
    prng = np.random.RandomState(seed)

    X = np.zeros((K * nPerState, D))
    Z = np.zeros(K * nPerState)
    for k in range(K):
        start = k * nPerState
        stop = (k + 1) * nPerState

        X[start:stop] = prng.rand(nPerState, D) < phi[k][np.newaxis, :]
        Z[start:stop] = k * np.ones(nPerState)
    return X, Z


if __name__ == '__main__':
    from matplotlib import pylab
    rcParams = pylab.rcParams
    rcParams['ps.fonttype'] = 42
    rcParams['ps.useafm'] = True
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['font.size'] = 25
    X, Z = genToyData()

    pylab.imshow(X, aspect=X.shape[1] / float(X.shape[0]),
                 interpolation='nearest', cmap='bone')
    pylab.show(block=True)
