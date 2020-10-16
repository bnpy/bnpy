'''
ToyARK13.py

Toy data from a first-order auto-regressive process, with D=3 dimensions.

There are K=13 states, including
* one pure high-noise state, with no time dynamics
* one set of 3 states produces "clock-wise" circles with stationary dim 0
* one set of 3 states produces "ctr-clock-wise" circles with stationary dim 0
* one set of 3 states produces "clock-wise" circles with stationary dim 1
* one set of 3 states produces "ctr-clock-wise" circles with stationary dim 1

Each set of 3 states varies the speed of rotation and scale of noise,
from slower and more noisy to faster and less noisy.

Usage
---------
To take a look at the states, just run this script as an executable.
$ python ToyARK13.py
'''
import numpy as np
import scipy.linalg

from bnpy.data import GroupXData


def get_short_name():
    ''' Return short string used in filepaths to store solutions
    '''
    return 'ToyARK13'


def get_data_info():
    return 'Toy AutoRegressive Data. %d true clusters.' % (K)


def get_data(seed=8675309, nDocTotal=52, T=800, **kwargs):
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
    X, Xprev, TrueZ, doc_range = genToyData(
        seed=seed, nDocTotal=nDocTotal, T=T)
    Data = GroupXData(X=X, TrueZ=TrueZ, Xprev=Xprev, doc_range=doc_range)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


# Make A in 3D
#######################################################

def makeA_3DRotationMatrix(degPerStep, stationaryDim):
    theta = degPerStep * np.pi / 180.  # radPerStep
    A = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    A3D = np.zeros((3, 3))
    activeDims = np.setdiff1d([0, 1, 2], [stationaryDim])
    for dorig, d in enumerate(activeDims):
        A3D[d, activeDims] = A[dorig, :]
    return A3D

K = 13
D = 3
degPerSteps = [10, 15, 20]
sigma2s = [.005, 0.001, 0.0005]

A = np.zeros((K, D, D))
Sigma = np.zeros((K, D, D))

# 3 states with clockwise rotation, around x axis
stationaryDim = 0
for kk in range(3):
    A[kk] = makeA_3DRotationMatrix(degPerSteps[kk], stationaryDim)
    Sigma[kk] = sigma2s[kk] * np.eye(D)
# 3 states with counter-clockwise rotation
for kk in range(3):
    A[5 - kk] = makeA_3DRotationMatrix(-1 * degPerSteps[kk], stationaryDim)
    Sigma[5 - kk] = sigma2s[kk] * np.eye(D)

# 3 states with clockwise rotation, around y axis
stationaryDim = 1
for kk in range(3):
    A[6 + kk] = makeA_3DRotationMatrix(degPerSteps[kk], stationaryDim)
    Sigma[6 + kk] = sigma2s[kk] * np.eye(D)
# 3 states with counter-clockwise rotation
for kk in range(3):
    A[11 - kk] = makeA_3DRotationMatrix(-1 * degPerSteps[kk], stationaryDim)
    Sigma[11 - kk] = sigma2s[kk] * np.eye(D)

A[-1] = np.zeros((D, D))
Sigma[-1] = 100 * np.max(sigma2s) * np.eye(D)

transPi = np.asarray([
    [.998, .002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, .998, .002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, .998, 0, 0, 0, 0, 0, 0, 0, 0, 0, .002],
    [0, 0, 0, .998, 0, 0, 0, 0, 0, 0, 0, 0, .002],
    [0, 0, 0, .002, .998, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, .002, .998, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, .998, .002, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, .998, .002, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, .998, 0, 0, 0, .002],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, .998, 0, 0, .002],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, .002, .998, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .002, .998, 0],
    [.003, 0, 0, 0, 0, .003, .003, 0, 0, 0, 0, .003, .988],
])
startStates = [0, 5, 6, 11]

cholSigma = np.zeros_like(Sigma)
for k in range(K):
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k])


def genToyData(seed=1234, nDocTotal=52, T=800):
    ''' TODO
    '''
    nDocTotal = int(nDocTotal)
    T = int(T)

    prng = np.random.RandomState(seed)
    states0toKm1 = np.arange(K)

    doc_range = np.zeros(nDocTotal + 1, dtype=np.int32)
    for i in range(1, nDocTotal + 1):
        doc_range[i] = doc_range[i - 1] + T

    N = doc_range[-1]
    allX = np.zeros((N, D))
    allXprev = np.zeros((N, D))
    allZ = np.zeros(N, dtype=np.int32)

    # Each iteration generates one time-series/sequence
    # with starting state deterministically rotating among all states
    for i in range(nDocTotal):
        start = doc_range[i]
        stop = doc_range[i + 1]

        T = stop - start
        Z = np.zeros(T + 1)
        X = np.zeros((T + 1, D))
        Z[0] = startStates[i % len(startStates)]
        X[0] = np.ones(D)
        nConsec = 0
        for t in range(1, T + 1):
            transPi_t = transPi[Z[t - 1]].copy()
            if nConsec > 120:
                transPi_t[Z[t - 1]] = 0
                transPi_t /= transPi_t.sum()

            Z[t] = prng.choice(states0toKm1, p=transPi_t)
            X[t] = prng.multivariate_normal(np.dot(A[Z[t]], X[t - 1]),
                                            Sigma[Z[t]])
            if Z[t] == Z[t - 1]:
                nConsec += 1
            else:
                nConsec = 0

        allZ[start:stop] = Z[1:]
        allX[start:stop] = X[1:]
        allXprev[start:stop] = X[:-1]

    return allX, allXprev, allZ, doc_range


def plotSequenceForRotatingState(degPerStep, Sigma, T=1000):
    theta = degPerStep * np.pi / 180.  # radPerStep
    A = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    Sigma = np.asarray(Sigma)
    if Sigma.size == 1:
        Sigma = Sigma * np.eye(2)
    elif Sigma.size == 2:
        Sigma = np.diag(Sigma)

    X = np.zeros((T, 2))
    X[0, :] = [1, 0]
    for t in range(1, T):
        X[t] = np.random.multivariate_normal(np.dot(A, X[t - 1]), Sigma)

    pylab.plot(X[:, 0], X[:, 1], '.')
    pylab.axis([-4, 4, -4, 4])
    pylab.axis('equal')


def plotSequenceForRotatingState3D(degPerStep, Sigma, stationaryDim=0, T=1000):
    A = makeA_3DRotationMatrix(degPerStep, stationaryDim)
    Sigma = Sigma * np.eye(3)
    assert Sigma.shape == (3, 3)

    X = np.zeros((T, 3))
    X[0, :] = [1, 1, 1]
    for t in range(1, T):
        X[t] = np.random.multivariate_normal(np.dot(A, X[t - 1]), Sigma)

    ax = Axes3D(pylab.figure())
    pylab.plot(X[:, 0], X[:, 1], X[:, 2], '.')
    pylab.axis('equal')


def showEachSetOfStatesIn3D():
    ''' Make a 3D plot in separate figure for each of the 3 states in a "set"

        These three states just vary the speed of rotation and scale of noise,
        from slow and large to fast and smaller.
    '''
    from matplotlib import pylab
    from mpl_toolkits.mplot3d import Axes3D
    L = len(degPerSteps)
    for ii in range(L):
        plotSequenceForRotatingState3D(-1 * degPerSteps[ii], sigma2s[ii], 2)


BlueSet = ['#253494',
           '#2c7fb8',
           '#41b6c4',
           ]
GreenSet = ['#006d2c',
            '#2ca25f',
            '#66c2a4',
            ]
RedSet = ['#b30000',
          '#e34a33',
          '#fc8d59',
          ]
PurpleSet = ['#7a0177',
             '#c51b8a',
             '#f768a1',
             ]

Colors = BlueSet + GreenSet[::-1] + RedSet + PurpleSet[::-1]
Colors.append('#969696')

if __name__ == '__main__':
    from matplotlib import pylab
    rcParams = pylab.rcParams
    rcParams['ps.fonttype'] = 42
    rcParams['ps.useafm'] = True
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['font.size'] = 25
    X, Xprev, Z, doc_range = genToyData(nDocTotal=12)

    N = doc_range.size - 1
    N = np.minimum(N, 4)

    ylabels = ['x', 'y', 'z']
    for dim in [0, 1, 2]:
        pylab.subplots(nrows=N, ncols=1, figsize=(6, 4))
        for n in range(N):
            start = doc_range[n]
            stop = doc_range[n + 1]
            X_n = X[start:stop]
            Z_n = Z[start:stop]

            pylab.subplot(N, 1, n + 1)
            pylab.hold('on')
            for k in range(K):
                Z_n_eq_k = np.flatnonzero(Z_n == k)
                pylab.plot(Z_n_eq_k, X_n[Z_n_eq_k, dim], '.', color=Colors[k])
            pylab.ylim([-2, 2])
            pylab.yticks([-1, 1])
            if n == 0:
                pylab.title(ylabels[dim])
            if n < N - 1:
                pylab.xticks([])
            else:
                pylab.xticks([200, 400, 600, 800])
        pylab.subplots_adjust(bottom=0.14)
        pylab.savefig('DataIllustration-ToyARK13-%s.eps' % (ylabels[dim]),
                      bbox_inches='tight', pad_inches=0)

    N = np.zeros(K)
    for k in range(K):
        N[k] = np.sum(Z == k)
    # print ['%4d ' % (N[k]) for k in xrange(K)]
    pylab.show(block=True)
