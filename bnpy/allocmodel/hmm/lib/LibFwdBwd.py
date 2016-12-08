import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
from bnpy.util import as2D

def cppReady():
    ''' Returns true if compiled cpp library available, false o.w.
    '''
    return hasEigenLibReady

def FwdAlg_cpp(initPi, transPi, SoftEv, order='C'):
    ''' Forward algorithm for a single HMM sequence. Implemented in C++/Eigen.
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    T, K = SoftEv.shape
    # Prep inputs
    initPi = np.asarray(initPi, order=order)
    transPi = np.asarray(transPi, order=order)
    SoftEv = np.asarray(SoftEv, order=order)

    # Allocate outputs
    fwdMsg = np.zeros((T, K), order=order)
    margPrObs = np.zeros(T, order=order)

    # Execute C++ code (fills in outputs in-place)
    lib.FwdAlg(initPi, transPi, SoftEv, fwdMsg, margPrObs, K, T)
    return fwdMsg, margPrObs


def BwdAlg_cpp(initPi, transPi, SoftEv, margPrObs, order='C'):
    ''' Backward algorithm for a single HMM sequence. Implemented in C++/Eigen.
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")

    # Prep inputs
    T, K = SoftEv.shape
    initPi = np.asarray(initPi, order=order)
    transPi = np.asarray(transPi, order=order)
    SoftEv = np.asarray(SoftEv, order=order)
    margPrObs = np.asarray(margPrObs, order=order)

    # Allocate outputs
    bMsg = np.zeros((T, K), order=order)

    # Execute C++ code for backward pass (fills in bMsg in-place)
    lib.BwdAlg(initPi, transPi, SoftEv, margPrObs, bMsg, K, T)
    return bMsg


def SummaryAlg_cpp(initPi, transPi, SoftEv, margPrObs, fMsg, bMsg,
                   mPairIDs=None,
                   order='C'):
    ''' Backward algorithm for a single HMM sequence. Implemented in C++/Eigen.
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")

    # Prep inputs
    T, K = SoftEv.shape
    initPi = np.asarray(initPi, order=order)
    transPi = np.asarray(transPi, order=order)
    SoftEv = np.asarray(SoftEv, order=order)
    margPrObs = np.asarray(margPrObs, order=order)
    fMsg = np.asarray(fMsg, order=order)
    bMsg = np.asarray(bMsg, order=order)

    if mPairIDs is None or len(mPairIDs) == 0:
        M = 0
        mPairIDs = np.zeros((0, 2))
    else:
        mPairIDs = as2D(np.asarray(mPairIDs, dtype=np.float64))
        M = mPairIDs.shape[0]
    assert mPairIDs.shape[0] == M
    assert mPairIDs.shape[1] == 2

    # Allocate outputs
    TransStateCount = np.zeros((K, K), order=order)
    Htable = np.zeros((K, K), order=order)
    mHtable = np.zeros((2 * M, K), order=order)

    # Execute C++ code for backward pass (fills in bMsg in-place)
    lib.SummaryAlg(initPi, transPi, SoftEv, margPrObs, fMsg, bMsg,
                   TransStateCount, Htable, mPairIDs, mHtable, K, T, M)
    return TransStateCount, Htable, mHtable

''' This block of code loads the shared library and defines wrapper functions
    that can take numpy array objects.
'''
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libfilename = 'libfwdbwdcpp.so'
hasEigenLibReady = True

try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename))
    lib.FwdAlg.restype = None
    lib.FwdAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

    lib.BwdAlg.restype = None
    lib.BwdAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

    lib.SummaryAlg.restype = None
    lib.SummaryAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int, ctypes.c_int]


except OSError:
    # No compiled C++ library exists
    hasEigenLibReady = False
