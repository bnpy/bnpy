'''
RCToyHMM: Reverse-cycles toy dataset

From Foti et al. "Stochastic Variational inference for Hidden Markov Models"
'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.viz import GaussViz


def get_data(seed=123456, nDocTotal=32, T=1000, **kwargs):
    ''' Generate several data sequences, returned as a bnpy data-object

    Args
    -------
    seed : integer seed for random number generator,
          used for actually *generating* the data
    seqLens : total number of observations in each sequence

    Returns
    -------
    Data : bnpy GroupXData object, with nObsTotal observations
    '''
    fullX, fullZ, doc_range = get_X(seed, nDocTotal, T)
    X = np.vstack(fullX)
    Z = np.asarray(fullZ)

    nUsedStates = len(np.unique(Z))
    if nUsedStates < K:
        print('WARNING: NOT ALL TRUE STATES USED IN GENERATED DATA')

    Data = GroupXData(X=X, doc_range=doc_range, TrueZ=Z)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_short_name():
    return 'RCToyHMM'


def get_data_info():
    return 'Toy HMM data with reverse-cycle transition matrix.'


D = 2
K = 8
initPi = 1.0 / K * np.ones(K)
transPi = np.asarray([
    [.01, .99, 0, 0, 0, 0, 0, 0],
    [0, .01, .99, 0, 0, 0, 0, 0],
    [.85, 0, 0, .15, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, .01, .99, 0, 0],
    [0, 0, 0, 0, 0, .01, .99, 0],
    [0, 0, 0, 0, .85, 0, 0, .15],
    [1, 0, 0, 0, 0, 0, 0, 0],
])

# Means for each component
mus = np.asarray([
    [-50, 0],
    [30, -30],
    [30, 30],
    [-100, -10],
    [40, -40],
    [-65, 0],
    [40, 40],
    [100, 10],
])

# Covariance for each component
# set to 20 times the 2x2 identity matrix
sigmas = np.tile(20 * np.eye(2), (K, 1, 1))


def get_X(seed, nDocTotal=None, T=None):
    ''' TODO
    '''
    nDocTotal = int(nDocTotal)
    T = int(T)

    prng = np.random.RandomState(seed)

    fullX = list()
    fullZ = list()
    doc_range = np.zeros(nDocTotal + 1, np.int32)

    # Each iteration generates one time-series/sequence
    # with starting state deterministically rotating among all states
    for i in range(nDocTotal):
        Z = list()
        X = list()
        initState = i % K
        initX = prng.multivariate_normal(mus[initState, :],
                                         sigmas[initState, :, :])
        Z.append(initState)
        X.append(initX)
        for j in range(T - 1):
            nextState = prng.choice(range(K), p=transPi[Z[j]])

            nextX = prng.multivariate_normal(mus[nextState, :],
                                             sigmas[nextState, :, :])
            Z.append(nextState)
            X.append(nextX)

        fullZ = np.hstack([fullZ, Z])  # need to concatenate as 1D
        fullX.append(X)
        doc_range[i + 1] = doc_range[i] + T

    return (np.vstack(fullX),
            np.asarray(fullZ, dtype=np.int32).flatten(),
            doc_range,
            )
