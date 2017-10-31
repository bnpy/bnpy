from builtins import *
import numpy as np
EPS = 1e-8


def create_initrho(K):
    ''' Make vector rho that implies E[beta] is nearly uniform.

    Returns
    --------
    rho : 1D array, size K
        Each entry rho[k] >= 0.

    Post Condition
    -------
    E[leftover mass] = r
    E[beta_k] \approx (1-r)/K
        where r is a small amount of remaining/leftover mass
    '''
    remMass = np.minimum(0.1, 1.0 / (K * K))
    # delta = 0, -1 + r, -2 + 2r, ...
    delta = (-1 + remMass) * np.arange(0, K, 1, dtype=np.float)
    rho = (1 - remMass) / (K + delta)
    return rho


def create_initomega(K, nDoc, gamma):
    ''' Make initial guess for omega.
    '''
    return (nDoc / K + gamma) * np.ones(K)


def forceRhoInBounds(rho, EPS=EPS):
    ''' Verify every entry of rho lies within [EPS, 1-EPS]

    Guarantees numerical stability.

    Returns
    -------
    rho : 1D array, size K
        Each entry satisfies: EPS <= rho[k] <= 1-EPS.
    '''
    rho = np.maximum(np.minimum(rho, 1.0 - EPS), EPS)
    return rho


def forceOmegaInBounds(omega, EPS=EPS, maxOmegaVal=None, Log=None):
    ''' Verify every entry of omega is bigger than EPS

    Returns
    -------
    omega : 1D array, size K
        Each entry satisfies: EPS <= omega[k]
    '''
    if Log is not None:
        nUp = np.sum(omega < EPS)
        if nUp > 0:
            Log.error("Forcing %d omega entries above minOmegaVal=%.3e." % (
                nUp, EPS))
        if maxOmegaVal is not None:
            nDown = np.sum(omega > maxOmegaVal)
            if nDown > 0:
                Log.error("Forcing %d omega entries below maxOmegaVal=%.3e" % (
                    nDown, maxOmegaVal))

    np.maximum(omega, EPS, out=omega)
    if maxOmegaVal is not None:
        np.minimum(omega, maxOmegaVal, out=omega)
    return omega


def rho2beta_active(rho):
    ''' Calculate probability vector for all active components.

    Returns
    --------
    beta : 1D array, size K
        beta[k] := probability of topic k
        Will have positive entries whose sum is <= 1.
    '''
    rho = np.asarray(rho, dtype=np.float64)
    beta = rho.copy()
    beta[1:] *= np.cumprod(1 - rho[:-1])
    return beta


def rho2beta(rho, returnSize='K+1'):
    ''' Calculate probability for all components including remainder.

    Returns
    --------
    beta : 1D array, size equal to 'K' or 'K+1', depending on returnSize
        beta[k] := probability of topic k
    '''
    rho = np.asarray(rho, dtype=np.float64)
    if returnSize == 'K':
        beta = rho.copy()
        beta[1:] *= np.cumprod(1 - rho[:-1])
    else:
        beta = np.append(rho, 1.0)
        beta[1:] *= np.cumprod(1.0 - rho)
    return beta


def beta2rho(beta, K):
    ''' Get rho vector that can deterministically produce provided beta.

    Returns
    ------
    rho : 1D array, size K
        Each entry rho[k] >= 0.
    '''
    beta = np.asarray(beta, dtype=np.float64)
    rho = beta.copy()
    beta_gteq = 1 - np.cumsum(beta[:-1])
    rho[1:] /= np.maximum(1e-100, beta_gteq)
    if beta.size == K + 1:
        return rho[:-1]
    elif beta.size == K:
        return rho
    else:
        raise ValueError('Provided beta needs to be of length K or K+1')


def sigmoid(c):
    ''' Calculates the sigmoid function at each entry of provided array.

    sigmoid(c) = 1./(1+exp(-c))

    Parameters
    -------
    c : array_like

    Returns
    ------
    v : array of same size as c
        v[k] = sigmoid(c[k])
        Satisfies 0 <= v[k] <= 1

    Notes
    -------
    Automatically enforces result away from "boundaries" [0, 1]
    This step is crucial to avoid overflow/NaN problems in optimization
    '''
    v = 1.0 / (1.0 + np.exp(-c))
    v = np.minimum(np.maximum(v, EPS), 1 - EPS)
    return v


def invsigmoid(v):
    ''' Get the inverse of the sigmoid function at each entry of array.

    Args
    --------
    v : array_like
        Each entry satisifies 0 < v[k] < 1

    Returns
    -------
    c : array_like, size of v
        c[k] = invsigmoid(v[k]), or  v[k] = sigmoid(c[k]).
    '''
    assert np.max(v) <= 1 - EPS
    assert np.min(v) >= EPS
    return -np.log((1.0 / v - 1))
