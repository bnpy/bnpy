'''
LibRlogR.py

Library of routines for computing assignment entropies for merges in C/C++

Notes
-------
Eigen expects the matrices to be fortran formated (column-major ordering).
In contrast, Numpy defaults to C-format (row-major ordering)
All functions here take care of this under the hood
(so end-user doesn't need to worry about alignment)
This explains the mysterious line: X=np.asarray(X, order='F')
However, we do *return* values that are F-ordered by default.
'''
from builtins import *
import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

# Compiled library is in the lib/ directory
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libpath = os.path.join(libpath, 'lib')
doUseLib = True
try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, 'librlogr.so'))
    lib.CalcRlogR_AllPairs.restype = None
    lib.CalcRlogR_AllPairs.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

    lib.CalcRlogR_AllPairsDotV.restype = None
    lib.CalcRlogR_AllPairsDotV.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

    lib.CalcRlogR_SpecificPairsDotV.restype = None
    lib.CalcRlogR_SpecificPairsDotV.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int, ctypes.c_int]

except OSError:
    # No compiled C++ library exists
    doUseLib = False

# RlogR all-pairs
###########################################################


def calcRlogR_allpairs_c(R):
    if not doUseLib:
        raise NotImplementedError(
            'LibRlogR library not found. Please recompile.')
    R = np.asarray(R, order='F')
    N, K = R.shape
    Z = np.zeros((K, K), order='F')
    lib.CalcRlogR_AllPairs(R, Z, N, K)
    return Z


# RlogR all-pairs
# with vector
def calcRlogRdotv_allpairs_c(R, v):
    if not doUseLib:
        raise NotImplementedError(
            'LibRlogR library not found. Please recompile.')
    R = np.asarray(R, order='F')
    v = np.asarray(v, order='F')
    N, K = R.shape
    Z = np.zeros((K, K), order='F')
    lib.CalcRlogR_AllPairsDotV(R, v, Z, N, K)
    return Z


# RlogR with vector
# specific pairs

def calcRlogRdotv_specificpairs_c(R, v, mPairs):
    if not doUseLib:
        raise NotImplementedError(
            'LibRlogR library not found. Please recompile.')
    R = np.asarray(R, order='F')
    v = np.asarray(v, order='F')
    N, K = R.shape
    Z = np.zeros((K, K), order='F')
    if K == 1 or len(mPairs) == 0:
        return Z
    aList, bList = list(zip(*mPairs))
    avec = np.asarray(aList, order='F', dtype=np.float64)
    bvec = np.asarray(bList, order='F', dtype=np.float64)

    lib.CalcRlogR_SpecificPairsDotV(R, v, avec, bvec, Z, N, len(avec), K)
    return Z
