'''
LibSparseResp.py

Sets global variable "hasLibReady" with True/False indicator
for whether the compiled cpp library required has compiled and is loadable successfully.
'''
import os
import numpy as np
import ctypes
import scipy.sparse
import sys, sysconfig

from numpy.ctypeslib import ndpointer

''' This block of code loads the shared library and defines wrapper functions
    that can take numpy array objects.
'''
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libfilename = 'libsparsemix' + sysconfig.get_config_var('EXT_SUFFIX')
hasLibReady = False

try:
    # Load the compiled C++ library from disk
    libpath = os.path.join(libpath, libfilename)
    assert os.path.exists(libpath)
    lib = ctypes.cdll.LoadLibrary(libpath)

except AssertionError as e:
    hasLibReady = False
except OSError as e:
    # No compiled C++ library exists
    hasLibReady = False
else:
    # lib has been loaded correctly from shared library
    hasLibReady = True

    # Now specify each function's signature
    lib.sparsifyResp.restype = None
    lib.sparsifyResp.argtypes = \
        [ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

    lib.sparsifyLogResp.restype = None
    lib.sparsifyLogResp.argtypes = \
        [ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

    lib.calcRlogR_withSparseRespCSR.restype = None
    lib.calcRlogR_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcMergeRlogR_withSparseRespCSR.restype = ctypes.c_double
    lib.calcMergeRlogR_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ]

    lib.calcRXXT_withSparseRespCSR.restype = None
    lib.calcRXXT_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcRXX_withSparseRespCSR.restype = None
    lib.calcRXX_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcRXX_withSparseRespCSC.restype = None
    lib.calcRXX_withSparseRespCSC.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]


    lib.calcRlogRdotv_withSparseRespCSR.restype = None
    lib.calcRlogRdotv_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcMergeRlogRdotv_withSparseRespCSR.restype = ctypes.c_double
    lib.calcMergeRlogRdotv_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ]


    lib.calcRXX_withSparseRespCSC.restype = None
    lib.calcRXX_withSparseRespCSC.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]


def sparsifyResp_cpp(Resp, nnzPerRow, order='C'):
    '''
    '''
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("Provided array must have row-major order.")
    N, K = Resp.shape

    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        spR_colids = np.argmax(Resp, axis=1)
        spR_data = np.ones(N, dtype=np.float64)
    else:
        # Prep input to C++ routine. Verify correct byte-order (row-major).
        Resp = np.asarray(Resp, order=order)
        # Allocate output arrays, initialized to all zeros
        spR_data = np.zeros(N * nnzPerRow, dtype=np.float64, order=order)
        spR_colids = np.zeros(N * nnzPerRow, dtype=np.int32, order=order)
        # Execute C++ code (fills in outputs in-place)
        lib.sparsifyResp(Resp, nnzPerRow, N, K, spR_data, spR_colids)


    # Here, both spR_data and spR_colids have been created
    # Assemble these into a row-based sparse matrix (scipy object)
    spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow,
                           step=nnzPerRow, dtype=spR_colids.dtype)
    spR = scipy.sparse.csr_matrix(
        (spR_data, spR_colids, spR_indptr),
        shape=(N,K),
        )
    return spR

def sparsifyLogResp_cpp(logResp, nnzPerRow, order='C'):
    ''' Compute sparse resp from log weights

    Example
    -------
    >>> from bnpy.util.SparseRespUtil import sparsifyLogResp
    >>> from bnpy.util.SparseRespUtil import sparsifyLogResp_numpy_vectorized

    >>> logResp = np.asarray([-1.0, -2, -3, -4, -100, -200])
    >>> if hasLibReady: spR = sparsifyLogResp_cpp(logResp[np.newaxis,:], 2)
    >>> if not hasLibReady: spR = sparsifyLogResp(logResp[np.newaxis,:], 2)
    >>> print(spR.data.sum())
    1.0
    >>> print(spR.indices.min())
    0
    >>> print(spR.indices.max())
    1
    >>> print(spR.toarray())
    [[0.73105858 0.26894142 0.         0.         0.         0.        ]]

    >>> # Try duplicates in weights that don't influence top L
    >>> logResp = np.asarray([-500., -500., -500., -4, -1, -2])
    >>> if hasLibReady: spR = sparsifyLogResp_cpp(logResp[np.newaxis,:], 3)
    >>> if not hasLibReady: spR = sparsifyLogResp(logResp[np.newaxis,:], 3)

    >>> print(spR.data.sum())
    1.0
    >>> print(np.unique(spR.indices))
    [3 4 5]

    >>> # Try duplicates in weights that DO influence top L
    >>> logResp = np.asarray([-500., -500., -500., -500., -1, -2])
    >>> if hasLibReady: spR = sparsifyLogResp_cpp(logResp[np.newaxis,:], 4)
    >>> if not hasLibReady: spR = sparsifyLogResp(logResp[np.newaxis,:], 4)
    >>> print(spR.data.sum())
    1.0
    >>> print(np.unique(spR.indices))
    [2 3 4 5]

    >>> # Try big problem
    >>> logResp = np.log(np.random.rand(100, 10))
    >>> if hasLibReady: spR_cpp = sparsifyLogResp_cpp(logResp, 7)
    >>> if not hasLibReady: spR_cpp = sparsifyLogResp(logResp, 7)
    >>> spR_np = sparsifyLogResp_numpy_vectorized(logResp, 7)
    >>> np.allclose(spR_cpp.toarray(), spR_np.toarray())
    True

    Returns
    -------
    spR : csr_matrix
    '''
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = logResp.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        spR_colids = np.argmax(logResp, axis=1)
        spR_data = np.ones(N, dtype=np.float64)
    else:
        # Prep input to C++ routine. Verify correct byte-order (row-major).
        logResp = np.asarray(logResp, order=order)
        # Allocate output arrays, initialized to all zeros
        spR_data = np.zeros(N * nnzPerRow, dtype=np.float64, order=order)
        spR_colids = np.zeros(N * nnzPerRow, dtype=np.int32, order=order)
        # Execute C++ code (fills in outputs in-place)
        lib.sparsifyLogResp(logResp, nnzPerRow, N, K, spR_data, spR_colids)

    # Here, both spR_data and spR_colids have been created
    # Assemble these into a row-based sparse matrix (scipy object)
    spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow,
                           step=nnzPerRow, dtype=spR_colids.dtype)
    spR = scipy.sparse.csr_matrix(
        (spR_data, spR_colids, spR_indptr),
        shape=(N,K),
        )
    return spR


def calcRlogR_withSparseRespCSR_cpp(
        spR_csr=None, nnzPerRow=-1, order='C', **kwargs):
    '''
    '''
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")

    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return 0.0
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        Hvec_OUT = np.zeros(K, dtype=np.float64)
        # Execute C++ code (fills in output array Hvec_OUT in-place)
        lib.calcRlogR_withSparseRespCSR(
            spR_csr.data,
            spR_csr.indices,
            spR_csr.indptr,
            K,
            N,
            nnzPerRow,
            Hvec_OUT)
        return Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))

def calcRlogRdotv_withSparseRespCSR_cpp(
        spR_csr=None, v=None, nnzPerRow=-1, order='C', **kwargs):
    '''
    '''
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    v = np.asarray(v, order=order)
    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return 0.0
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        Hvec_OUT = np.zeros(K, dtype=np.float64)
        # Execute C++ code (fills in output array Hvec_OUT in-place)
        lib.calcRlogRdotv_withSparseRespCSR(
            spR_csr.data,
            spR_csr.indices,
            spR_csr.indptr,
            v,
            K,
            N,
            nnzPerRow,
            Hvec_OUT)
        return Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))

def calcMergeRlogR_withSparseRespCSR_cpp(
        spR_csr=None, nnzPerRow=-1, order='C', mPairIDs=None, **kwargs):
    '''
    '''
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return None
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        m_Hvec_OUT = np.zeros(len(mPairIDs), dtype=np.float64)
        for mID, (kA, kB) in enumerate(mPairIDs):
            # Execute C++ code (fills in output array Hvec_OUT in-place)
            m_Hvec_OUT[mID] = lib.calcMergeRlogR_withSparseRespCSR(
                spR_csr.data,
                spR_csr.indices,
                spR_csr.indptr,
                K,
                N,
                nnzPerRow,
                kA, kB)
        return m_Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))

def calcMergeRlogRdotv_withSparseRespCSR_cpp(
        spR_csr=None, nnzPerRow=-1, v=None,
        order='C', mPairIDs=None, **kwargs):
    '''
    '''
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return None
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        m_Hvec_OUT = np.zeros(len(mPairIDs), dtype=np.float64)
        for mID, (kA, kB) in enumerate(mPairIDs):
            # Execute C++ code (fills in output array Hvec_OUT in-place)
            m_Hvec_OUT[mID] = lib.calcMergeRlogRdotv_withSparseRespCSR(
                spR_csr.data,
                spR_csr.indices,
                spR_csr.indptr,
                v,
                K,
                N,
                nnzPerRow,
                kA, kB)
        return m_Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))


def calcRXXT_withSparseRespCSR_cpp(
        X=None, spR_csr=None, order='C', **kwargs):
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = spR_csr.shape
    N1, D = X.shape
    assert N == N1
    nnzPerRow = spR_csr.data.size // N
    X = np.asarray(X, order=order)
    stat_RXX = np.zeros((K, D, D), order=order)
    lib.calcRXXT_withSparseRespCSR(
        X, spR_csr.data, spR_csr.indices, spR_csr.indptr,
        D, K, N, nnzPerRow,
        stat_RXX)
    return stat_RXX


def calcRXX_withSparseRespCSC_cpp(
        X=None, spR_csc=None, order='C', **kwargs):
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = spR_csc.shape
    N1, D = X.shape
    assert N == N1
    L = spR_csc.data.size

    X = np.asarray(X, order=order)

    stat_RXX = np.zeros((K, D), order=order)

    lib.calcRXX_withSparseRespCSC(
        X, spR_csc.data, spR_csc.indices, spR_csc.indptr,
        D, K, L, N,
        stat_RXX)
    return stat_RXX

def calcRXX_withSparseRespCSR_cpp(
        X=None, spR_csr=None, order='C', **kwargs):
    if not hasLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = spR_csr.shape
    N1, D = X.shape
    assert N == N1
    nnzPerRow = spR_csr.data.size // N

    X = np.asarray(X, order=order)
    stat_RXX = np.zeros((K, D), order=order)

    lib.calcRXX_withSparseRespCSR(
        X, spR_csr.data, spR_csr.indices, spR_csr.indptr,
        D, K, N, nnzPerRow,
        stat_RXX)
    return stat_RXX




if __name__ == "__main__":
    from bnpy.util.SparseRespUtil import sparsifyResp_numpy_vectorized

    for nnzPerRow in [1, 2, 3]:
        R = np.random.rand(5,6)
        R /= R.sum(axis=1)[:,np.newaxis]
        print("input resp array")
        print("----------------")
        print(R)

        print("sparsifyResp_numpy_vectorized(R, nnzPerRow=%d)" % nnzPerRow)
        print("-----------------------------")
        spR2 = sparsifyResp_numpy_vectorized(R, nnzPerRow).toarray()
        print(spR2)

        print("sparsifyResp_cpp(R, nnzPerRow=%d)" % nnzPerRow)
        print("-----------------------------")
        spR = sparsifyResp_cpp(R, nnzPerRow).toarray()
        print(spR)

        assert np.allclose(spR, spR2)
