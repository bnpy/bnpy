"""
References
----------
Neal Hughes (no relation to Mike)
Blog post on "Fast Python loops with Cython"
http://nealhughes.net/cython1/
"""
from builtins import *
import numpy as np
from libc.math cimport log


def calcRlogR_1D_cython(double[:] R):
    """ Compute sum of R * log(R). Faster, cython version.

    Args
    ----
    R : 1D array, size N
        Each row must have entries that are strictly positive (> 0).
        No bounds checking is enforced!

    Returns
    -------
    H : scalar float
        H = np.sum(R * log R)
    """
    assert R.ndim == 1
    cdef int N = R.shape[0]
    # H is a memoryview here
    # aka a low-level pointer to array-like object
    cdef double H = 0.0
    # Compute using loops (fast!)
    for n in range(N):
        H += R[n] * log(R[n])
    # Return the numpy array, not a memoryview
    return H

def calcRlogR_cython(double[:, :] R):
    """ Compute sum over columns of R * log(R). Faster, cython version.

    Args
    ----
    R : 2D array, N x K
        Each row must have entries that are strictly positive (> 0).
        No bounds checking is enforced!

    Returns
    -------
    H : 1D array, size K
        H[k] = np.sum(R[:,k] * log R[:,k])
    """
    cdef int N = R.shape[0]
    cdef int K = R.shape[1]
    # H is a memoryview here
    # aka a low-level pointer to array-like object
    cdef double[:] H = np.zeros(K)
    # Compute using loops (fast!)
    for n in range(N):
        for k in range(K):
            H[k] += R[n,k] * log(R[n,k])
    # Return the numpy array, not a memoryview
    return np.asarray(H)


def calcRlogRdotv_cython(double[:, :] R, double[:] v):
    """ Compute sum over columns of R * log(R) with weight vector v.

    Args
    ----
    R : 2D array, N x K
        Each row must have entries that are strictly positive (> 0).
        No bounds checking is enforced!

    v : 1D array, size N
        Weight vector for each row of R

    Returns
    -------
    H : 1D array, size K
        H[k] = np.inner(v, R[:,k] * log R[:,k])
    """
    cdef int N = R.shape[0]
    cdef int K = R.shape[1]
    # H is a memoryview here
    # aka a low-level pointer to array-like object
    cdef double[:] H = np.zeros(K)
    # Compute using loops (fast!)
    for n in range(N):
        for k in range(K):
            H[k] += v[n] * R[n,k] * log(R[n,k])
    # Return the numpy array, not a memoryview
    return np.asarray(H)


def calcRlogRdotv_1D_cython(double[:] R, double[:] v):
    """ Compute sum over R * log(R) with weight vector v.

    Args
    ----
    R : 1D array, size N
        Each row must have entries that are strictly positive (> 0).
        No bounds checking is enforced!

    Returns
    -------
    H : double
        H = np.sum(v * R * log R)
    """
    assert R.ndim == 1
    cdef int N = R.shape[0]
    assert N == v.size
    # H is a memoryview here
    # aka a low-level pointer to array-like object
    cdef double H = 0.0
    # Compute using loops (fast!)
    for n in range(N):
        H += v[n] * R[n] * log(R[n])
    # Return the numpy array, not a memoryview
    return H
