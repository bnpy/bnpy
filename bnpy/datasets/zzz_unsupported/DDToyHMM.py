'''
DDToyHMM: Diagonally-dominant toy dataset

From Foti et al. "Stochastic Variational inference for Hidden Markov Models"
'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.viz import GaussViz


def get_data(seed=123, nDocTotal=32, T=1000,
             **kwargs):
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
    fullX, fullZ, doc_range = get_X(seed, T, nDocTotal)
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
    return 'DDToyHMM'


def get_data_info():
    return 'Toy HMM data with diagonally-dominant transition matrix.'

D = 2
K = 8
initPi = 1.0 / K * np.ones(K)
transPi = np.asarray([
    [.999, .001, 0, 0, 0, 0, 0, 0],
    [0, .999, .001, 0, 0, 0, 0, 0],
    [0, 0, .999, .001, 0, 0, 0, 0],
    [0, 0, 0, .999, .001, 0, 0, 0],
    [0, 0, 0, 0, .999, .001, 0, 0],
    [0, 0, 0, 0, 0, .999, .001, 0],
    [0, 0, 0, 0, 0, 0, .999, .001],
    [.001, 0, 0, 0, 0, 0, 0, .999],
])

# Means for each component
mus = np.asarray([
    [0, 20],
    [20, 0],
    [-30, -30],
    [30, -30],
    [-20, 0],
    [0, -20],
    [30, 30],
    [-30, 30],
])

# Covariance for each component
# set to the 2x2 identity matrix
sigmas = np.tile(np.eye(2), (K, 1, 1))


def get_X(seed, T, nDocTotal):
    ''' Generates X, Z, seqInds
    '''
    T = int(T)
    nDocTotal = int(nDocTotal)

    prng = np.random.RandomState(seed)

    fullX = list()
    fullZ = list()
    doc_range = np.zeros(nDocTotal + 1, dtype=np.int32)
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

Colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
          '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']


def illustrate(Colors=Colors):
    if hasattr(Colors, 'colors'):
        Colors = Colors.colors

    from matplotlib import pylab
    rcParams = pylab.rcParams
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['text.usetex'] = False
    rcParams['xtick.labelsize'] = 20
    rcParams['ytick.labelsize'] = 20
    rcParams['legend.fontsize'] = 25

    import bnpy

    Data = get_data(T=1000, nDocTotal=8)
    for k in range(K):
        zmask = Data.TrueParams['Z'] == k
        pylab.plot(Data.X[zmask, 0], Data.X[zmask, 1], '.', color=Colors[k],
                   markeredgecolor=Colors[k],
                   alpha=0.4)

        sigEdges = np.flatnonzero(transPi[k] > 0.0001)
        for j in sigEdges:
            if j == k:
                continue
            dx = mus[j, 0] - mus[k, 0]
            dy = mus[j, 1] - mus[k, 1]
            pylab.arrow(mus[k, 0], mus[k, 1],
                        0.8 * dx,
                        0.8 * dy,
                        head_width=2, head_length=4,
                        facecolor=Colors[k], edgecolor=Colors[k])

            tx = 0 - mus[k, 0]
            ty = 0 - mus[k, 1]
            xy = (mus[k, 0] - 0.2 * tx, mus[k, 1] - 0.2 * ty)
            '''
            pylab.annotate( u'\\u27F2',
                      xy=(mus[k,0], mus[k,1]),
                     color=Colors[k],
                     fontsize=35,
                    )
            '''
            pylab.gca().yaxis.set_ticks_position('left')
            pylab.gca().xaxis.set_ticks_position('bottom')

            pylab.axis('image')
            pylab.ylim([-38, 38])
            pylab.xlim([-38, 38])


if __name__ == '__main__':
    illustrate()
    pylab.savefig('DatasetIllustration-DDToyHMM.eps', bbox_inches='tight',
                  pad_inches=0)
    pylab.show(block=True)

# P.arrow( x, y, dx, dy, **kwargs )
# P.arrow( 0.5, 0.8, 0.0, -0.2, fc="k", ec="k",
# head_width=0.05, head_length=0.1 )
