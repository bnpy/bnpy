'''
RandUtil.py

Utilities for sampling (pseudo) random numbers
'''
from builtins import *
import numpy as np


def choice(candidates, ps=None, randstate=np.random):
    ''' Choose one element at random from list of candidates.
        ps[k] gives probability of candidate k
        ps need not sum to one, but all entries must be positive
    '''
    if ps is None:
        N = len(candidates)
        ps = np.ones(N) / N
    totals = np.cumsum(ps)
    r = randstate.rand() * totals[-1]
    k = np.searchsorted(totals, r)
    return candidates[k]


def multinomial(Nsamp, ps, randstate=np.random):
    """ Draw Nsamp samples from multinomial with symbol probabilities ps

    Basically a thin wrapper around np.random.multinomial,
    which allows provided probabilities to be not normalized.

    Parameters
    --------
    Nsamp : int
        number of samples
    ps : 1D array, size K
        Each entry ps[k] must be >= 0.
        Specifies discrete probability distribution over K choices,
        up to a multiplicative constant.
    randstate : numpy RandomState
        Random number generator.

    Returns
    --------
    c : 1D array size K
        Each entry c[k] >= 0.
        This vector will sum to Nsamp: sum(c) == Nsamp.
    """
    return randstate.multinomial(Nsamp, ps / ps.sum())


def mvnrand(mu, Sigma, N=1, PRNG=np.random.RandomState()):
    if isinstance(PRNG, int):
        PRNG = np.random.RandomState(PRNG)
    return PRNG.multivariate_normal(mu, Sigma, (N))


def rotateCovMat(Sigma, theta=np.pi / 4):
    ''' Get covariance matrix rotated by theta radians.

    This rotation preserves the underlying eigen structure.

    Parameters
    ------
    Sigma : 2D array, size DxD
        symmetric and postiive definite
    theta : float
        specifies rotation angle in radians

    Returns
    -----
    SigmaRot : 2D array, size DxD
        symmetric and positive definite

    '''
    RotMat = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    RotMat = np.asarray(RotMat)
    Lam, V = np.linalg.eig(Sigma)
    Lam = np.diag(Lam)
    Vrot = np.dot(V, RotMat)
    return np.dot(Vrot, np.dot(Lam, Vrot.T))
