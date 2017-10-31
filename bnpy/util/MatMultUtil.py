'''
Efficient matrix multiplication subroutines.

Thin wrappers around BLAS implementations,
that make sure the best alignment and striding are used.

Notes
-------
Timing results on several machines:
- late 2011 macbook (with Intel CPU)
- 32-bit desktop (with AMD CPU, ~3GHz)
- 64-bit desktop (with AMD CPU, ~3GHz)

X = np.random.rand(1e6, 64)

Compare methods for computing X.T * X
      A | fblas.dgemm(1.0, X, X, trans_a=True)
      B | fblas.dgemm(1.0, X.T, X.T, trans_b=True)
      C | np.dot(X.T,X)
                   C         A        B
      macbook      1.46 s    1.20 s    0.69 s
32-bit desktop     1.67 s    1.45 s    0.58 s
64-bit desktop     1.39 s    1.2 s     0.45 s

Conclusion: method "B" is the best by far!
'''
from builtins import *
import numpy as np

try:
    import scipy.linalg.blas
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        # Scipy changed location of BLAS libraries in late 2012.
        # See http://github.com/scipy/scipy/pull/358
        fblas = scipy.linalg.blas._fblas
except:
    raise ImportError(
        "BLAS libraries for efficient matrix multiplication not found")


def dotATB(A, B):
    ''' Compute matrix product A.T * B
        using efficient BLAS routines (low-level machine code)
    '''
    if A.shape[1] > B.shape[1]:
        return fblas.dgemm(1.0, A, B, trans_a=True)
    else:
        return np.dot(A.T, B)


def dotABT(A, B):
    ''' Compute matrix product A* B.T
        using efficient BLAS routines (low-level machine code)
    '''
    if B.shape[0] > A.shape[0]:
        return fblas.dgemm(1.0, A, B, trans_b=True)
    else:
        return np.dot(A, B.T)


def dotATA(A):
    ''' Compute matrix product A.T * A
        using efficient BLAS routines (low-level machine code)
    '''
    return fblas.dgemm(1.0, A.T, A.T, trans_b=True)
