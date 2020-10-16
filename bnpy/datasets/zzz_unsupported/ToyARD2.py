'''
ToyARD2.py

Toy data from a first-order auto-regressive process.
'''
import numpy as np
import scipy.linalg

from bnpy.data import XData


def get_short_name():
    ''' Return short string used in filepaths to store solutions
    '''
    return 'ToyARD2'


def get_data_info():
    return 'Toy AutoRegressive Data. %d true clusters.' % (K)


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
    X, Xprev, TrueZ = genToyData(seed, nObsTotal)
    Data = XData(X=X, TrueZ=TrueZ, Xprev=Xprev)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data

K = 2
D = 2

a1 = 0.9995
A = np.zeros((K, D, D))
A[0] = np.asarray([[a1, 0], [0, 0]])
A[1] = np.asarray([[0, 0], [0, a1]])

s1 = 0.001
s2 = 0.003
Sigma = np.zeros((K, D, D))
Sigma[0] = np.diag([s1, s2])
Sigma[1] = np.diag([s2, s1])

cholSigma = np.zeros_like(Sigma)
for k in range(K):
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k])


def genToyData(seed=0, nObsTotal=25000):
    ''' Generate Toy Data
    '''
    switchProb = 0.005
    Xprev = np.zeros((nObsTotal, D))
    X = np.zeros((nObsTotal, D))
    Xprev[0] = [.05, -.05]

    PRNG = np.random.RandomState(seed)
    XX1 = np.dot(cholSigma[1].T, PRNG.randn(D, nObsTotal)).T
    PRNG = np.random.RandomState(seed + 1)
    XX0 = np.dot(cholSigma[0].T, PRNG.randn(D, nObsTotal)).T

    PRNG = np.random.RandomState(seed + 2)
    rs = PRNG.rand(nObsTotal)
    Z = np.ones(nObsTotal)
    for n in range(nObsTotal):
        if Z[n] == 1:
            X[n] = np.dot(A[1], Xprev[n]) + XX1[n]
        elif Z[n] == -1:
            X[n] = np.dot(A[0], Xprev[n]) + XX0[n]

        if n < nObsTotal - 1:
            Xprev[n + 1] = X[n]
            if rs[n] < switchProb:
                Z[n + 1] = -1 * Z[n]
            else:
                Z[n + 1] = Z[n]
    Z[Z < 0] = 0
    return X, Xprev, Z

if __name__ == '__main__':
    X, Xprev, Z = genToyData(seed=0, nObsTotal=5800)
    from matplotlib import pylab

    aIDs = np.flatnonzero(Z == 0)
    bIDs = np.flatnonzero(Z == 1)
    B = np.max(np.abs(X))

    pylab.subplot(2, 1, 1)
    pylab.plot(aIDs, X[aIDs, 0], 'r.')
    pylab.plot(bIDs, X[bIDs, 0], 'b.')
    pylab.ylim([-B, B])

    pylab.subplot(2, 1, 2)
    pylab.plot(aIDs, X[aIDs, 1], 'r.')
    pylab.plot(bIDs, X[bIDs, 1], 'b.')
    pylab.ylim([-B, B])

    pylab.tight_layout()
    pylab.show()
