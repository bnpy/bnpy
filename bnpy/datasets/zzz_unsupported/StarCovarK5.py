'''
StarCovarK5
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data import XData

# Set Toy Parameters
K = 5
D = 2

w = np.asarray([5., 4., 3., 2., 1.])
w = w / w.sum()

Mu = np.zeros((K, D))

# Create basic 2D cov matrix with major axis much longer than minor one
V = 1.0 / 16.0
SigmaBase = np.asarray([[V, 0], [0, V / 100.0]])

# Create several Sigmas by rotating this basic covariance matrix
Sigma = np.zeros((5, D, D))
for k in range(4):
    Sigma[k] = rotateCovMat(SigmaBase, k * np.pi / 4.0)

# Add final Sigma with large isotropic covariance
Sigma[4] = 4 * V * np.eye(D)

# Precompute cholesky decompositions
cholSigma = np.zeros(Sigma.shape)
for k in range(K):
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k])

# Module Util Fcns


def sample_data_from_comp(k, Nk, PRNG):
    return Mu[k, :] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk)).T


def get_short_name():
    ''' Return short string used in filepaths to store solutions
    '''
    return 'StarCovarK5'


def get_data_info():
    return 'Overlapping Star Toy Data. %d true clusters.' % (K)

# Generate the Data


def get_X(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    trueList = list()
    Npercomp = PRNG.multinomial(nObsTotal, w)
    X = list()
    for k in range(K):
        X.append(sample_data_from_comp(k, Npercomp[k], PRNG))
        trueList.append(k * np.ones(Npercomp[k]))
    X = np.vstack(X)
    TrueZ = np.hstack(trueList)
    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ

# User-facing accessors


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
    X, TrueZ = get_X(seed, nObsTotal)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data
