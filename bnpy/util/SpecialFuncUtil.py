'''
SpecialFuncUtil.py

Special mathematical functions, like multivariate gammaln.
'''
from builtins import *
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, digamma

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.)
LOGTWOPI = np.log(2. * np.pi)
EPS = 10 * np.finfo(float).eps

MVgCache = defaultdict(lambda: dict())


def MVgammaln(x, D):
    ''' Compute log of D-dimensional multivariate Gamma func. for input x

    Parameters
    ------
    x : 1D array, size D
    D : int

    Returns
    ------
    r : float
        value of multivariate gamma function at provided input

    Notes
    -------
    Caching gives big speedup!
    * caching : 208 sec for 5 iters of CGS on K=50, D=2 problem with N=10000
    * no cache : 300 sec
    '''
    try:
        return MVgCache[D][x]
    except KeyError:
        dvec = np.arange(1, D + 1)
        gammalnterm = gammaln(x + 0.5 * (1 - dvec)).sum()
        result = gammalnterm + 0.25 * D * (D - 1) * LOGPI
        MVgCache[D][x] = result
    return result


def MVdigamma(x, D):
    ''' Compute the first-derivative of MVgammaln function

    Parameters
    ------
    x : 1D array, size D
    D : int

    Returns
    ------
    dx : 1D array, size D
    '''
    return digamma(x + 0.5 * (1 - np.arange(1, D + 1))).sum()


def logsumexp(logA, axis=None):
    ''' Efficiently compute log(sum(exp(...))) for input matrix "logA"
        Computation is both vectorized and numerically stable.
    '''
    logA = np.asarray(logA)
    logAmax = logA.max(axis=axis)
    if axis is None:
        logA = logA - logAmax
    elif axis == 1:
        logA = logA - logAmax[:, np.newaxis]
    elif axis == 0:
        logA = logA - logAmax[np.newaxis, :]
    assert np.allclose(logA.max(), 0.0)
    logA = np.log(np.sum(np.exp(logA), axis=axis))
    return logA + logAmax
