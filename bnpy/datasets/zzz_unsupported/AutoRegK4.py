'''
AutoRegK4.py

A simple toy dataset that uses an autoregressive gaussian likelihood and
K = 4 state HMM allocation model.

The dataset can be vizualized by running python AutoRegK4.py from the command
line.
'''

import numpy as np
from bnpy.data import GroupXData
import scipy.io

# Transition matrix
transPi = np.asarray([[0.97, 0.01, 0.01, 0.01],
                      [0.01, 0.97, 0.01, 0.01],
                      [0.01, 0.01, 0.97, 0.01],
                      [0.01, 0.01, 0.01, 0.97]])

initState = 1

K = 4
D = 2

# Using the variables below, the autoregressive likelihood says:
#  x[n] = A*Xprev[n] + Normal(0, Sigma)
# Define linear scale parameters, A
a1 = 0.9995
A = np.zeros((K, D, D))
A[0] = np.asarray([[1, 0], [0, 0]])  # red
A[1] = np.asarray([[0, 0], [0, -1]])  # blue
A[2] = np.asarray([[0, 0], [0, 0]])  # green
A[3] = np.asarray([[1, 0], [0, 1]])  # yellow

# Define noise parameters, Sigma
s1 = 0.001
s2 = 0.003
Sigma = np.zeros((K, D, D))
Sigma[0] = np.diag([s1, s2])
Sigma[1] = np.diag([s2, s1])
Sigma[2] = np.diag([s2, s1])
Sigma[3] = np.diag([s2, s1])
cholSigma = np.zeros_like(Sigma)
for k in range(K):
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k])


def get_data(seed=8675309, seqLen=6000, **kwargs):
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
    X, Xprev, TrueZ = genToyData(seed, seqLen)
    T = TrueZ.size
    doc_range = np.asarray([0, T])

    Data = GroupXData(X=X, Xprev=Xprev, TrueZ=TrueZ, doc_range=doc_range)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_short_name():
    return 'AutoRegK4'


def get_data_info():
    return 'Toy Autoregressive gaussian data with K = 4 clusters.'


def genToyData(seed=0, T=6000):
    '''

        Returns
        -------
        X
        Xprev
        Z : 1D array, size T
    '''

    # Pre-generate the noise that will be added at each step
    PRNG = np.random.RandomState(seed)
    Noise = np.zeros((K, T + 1, D))
    for k in range(K):
        PRNG = np.random.RandomState(seed + k)
        Noise[k, :, :] = np.dot(cholSigma[k].T, PRNG.randn(D, T + 1)).T

    PRNG = np.random.RandomState(seed + K)
    Z = np.zeros(T + 1, dtype=np.int32)
    X = np.zeros((T + 1, D))

    Z[0] = 0
    X[0] = 0

    stateSpace = np.arange(K)
    for t in range(1, T + 1):
        Z[t] = PRNG.choice(stateSpace, p=transPi[Z[t - 1]])
        X[t] = np.dot(A[Z[t]], X[t - 1]) + Noise[Z[t], t]

    return X[1:].copy(), X[:-1].copy(), Z[1:]


if __name__ == '__main__':
    X, Xprev, Z = genToyData(seed=0, T=6000)
    from matplotlib import pylab

    IDs0 = np.flatnonzero(Z == 0)
    IDs1 = np.flatnonzero(Z == 1)
    IDs2 = np.flatnonzero(Z == 2)
    IDs3 = np.flatnonzero(Z == 3)
    B = np.max(np.abs(X))

    pylab.subplot(3, 1, 1)
    pylab.plot(IDs0, X[IDs0, 0], 'r.')
    pylab.plot(IDs1, X[IDs1, 0], 'b.')
    pylab.plot(IDs2, X[IDs2, 0], 'g.')
    pylab.plot(IDs3, X[IDs3, 0], 'y.')
    pylab.ylim([-B, B])

    pylab.subplot(3, 1, 2)
    pylab.plot(IDs0, X[IDs0, 1], 'r.')
    pylab.plot(IDs1, X[IDs1, 1], 'b.')
    pylab.plot(IDs2, X[IDs2, 1], 'g.')
    pylab.plot(IDs3, X[IDs3, 1], 'y.')
    pylab.ylim([-B, B])

    pylab.subplot(3, 1, 3)
    pylab.plot(X[IDs0, 0], X[IDs0, 1], 'r.', markeredgecolor='r')
    pylab.plot(X[IDs1, 0], X[IDs1, 1], 'b.', markeredgecolor='b')
    pylab.plot(X[IDs3, 0], X[IDs3, 1], 'y.', markeredgecolor='y')
    pylab.plot(X[IDs2, 0], X[IDs2, 1], 'g.', markeredgecolor='g')

    pylab.tight_layout()
    pylab.show()
