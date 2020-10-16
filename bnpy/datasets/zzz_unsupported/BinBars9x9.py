'''
BinBars9x9.py

Binary toy bars data, with a 9x9 grid,
so each observation is a vector of size 81.

There are K=18 true topics, one for each row/col of the grid.
'''
import numpy as np
from bnpy.data import XData
from bnpy.util import as1D

K = 18  # Number of topics
D = 81  # Vocabulary Size

Defaults = dict()
Defaults['nObsTotal'] = 500
Defaults['bgProb'] = 0.05
Defaults['fgProb'] = 0.90
Defaults['seed'] = 8675309


def get_data(**kwargs):
    ''' Create dataset as bnpy DataObj object.
    '''
    Data = generateRandomBinaryDataFromMixture(**kwargs)
    Data.name = 'BinBars9x9'
    Data.summary = get_data_info()
    return Data


def get_data_info():
    s = 'Binary Bars Data with %d true topics.' % (K)
    return s


def makePhi(fgProb=0.75, bgProb=0.05, **kwargs):
    ''' Make phi matrix that defines probability of each pixel.
    '''
    phi = bgProb * np.ones((K, np.sqrt(D), np.sqrt(D)))
    for k in range(K):
        if k < K / 2:
            rowID = k
            # Horizontal bars
            phi[k, rowID, :] = fgProb
        else:
            colID = k - K / 2
            phi[k, :, colID] = fgProb
    phi = np.reshape(phi, (K, D))
    return phi


def generateRandomBinaryDataFromMixture(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    phi = makePhi(**kwargs)
    nObsTotal = kwargs['nObsTotal']

    PRNG = np.random.RandomState(kwargs['seed'])

    # Select number of observations from each cluster
    beta = 1.0 / K * np.ones(K)
    if nObsTotal < 2 * K:
        # force examples from every cluster
        nPerCluster = np.ceil(nObsTotal / K) * np.ones(K)
    else:
        nPerCluster = as1D(PRNG.multinomial(nObsTotal, beta, size=1))
    nPerCluster = np.int32(nPerCluster)

    # Generate data from each cluster!
    X = np.zeros((nObsTotal, D))
    Z = np.zeros(nObsTotal, dtype=np.int32)
    start = 0
    for k in range(K):
        stop = start + nPerCluster[k]
        X[start:stop] = np.float64(
            PRNG.rand(nPerCluster[k], D) < phi[k, :][np.newaxis, :])
        Z[start:stop] = k
        start = stop

    TrueParams = dict()
    TrueParams['beta'] = beta
    TrueParams['phi'] = phi
    TrueParams['Z'] = Z
    return XData(X, TrueParams=TrueParams)

if __name__ == '__main__':
    import bnpy.viz.BernViz as BernViz

    Data = get_data(nObsTotal=K)
    BernViz.plotCompsAsSquareImages(Data.TrueParams['phi'])
    BernViz.plotDataAsSquareImages(
        Data, unitIDsToPlot=np.arange(K), doShowNow=1)
