from builtins import *
import argparse
import numpy as np
import scipy.sparse
import timeit
import time
import sys

from .SparseRespUtil import sparsifyResp
from bnpy.util import dotATA
from bnpy.util.EntropyUtil import calcRlogR
from bnpy.util.ShapeUtil import as1D, toCArray

hasCPPLib = True
try:
    from .lib.sparseResp.LibSparseResp \
        import calcRlogRdotv_withSparseRespCSR_cpp
    from .lib.sparseResp.LibSparseResp import calcRlogR_withSparseRespCSR_cpp
    from .lib.sparseResp.LibSparseResp import calcRXXT_withSparseRespCSR_cpp
    from .lib.sparseResp.LibSparseResp import calcRXX_withSparseRespCSR_cpp
    from .lib.sparseResp.LibSparseResp import calcRXX_withSparseRespCSC_cpp

    from .lib.sparseResp.LibSparseResp \
        import calcMergeRlogR_withSparseRespCSR_cpp as calcSparseMergeRlogR
    from .lib.sparseResp.LibSparseResp \
        import calcMergeRlogRdotv_withSparseRespCSR_cpp as calcSparseMergeRlogRdotv
except ImportError:
    hasCPPLib = False
    # Sketchy avoid import errors
    calcSparseMergeRlogR = None
    calcSparseMergeRlogRdotv = None

def calcSparseRlogR(spR=None, nnzPerRow=-1, **kwargs):
    ''' Compute assignment entropy of each state.

    Returns
    -------
    H : 1D array, size K
    '''
    if not hasCPPLib:
        raise ValueError("Required compiled C++ library not found.")
    return calcRlogR_withSparseRespCSR_cpp(spR_csr=spR, nnzPerRow=nnzPerRow)

def calcSparseRlogRdotv(spR=None, v=None, nnzPerRow=-1, **kwargs):
    ''' Compute assignment entropy of each state.

    Returns
    -------
    H : 1D array, size K
    '''
    if not hasCPPLib:
        raise ValueError("Required compiled C++ library not found.")
    return calcRlogRdotv_withSparseRespCSR_cpp(
        spR_csr=spR, v=v, nnzPerRow=nnzPerRow)

def calcSpRXX(X=None, spR_csr=None, **kwargs):
    ''' Compute expected sum-of-squares statistic for each state.

    Returns
    -------
    S_xx : 3D array, K x D
    '''
    if not hasCPPLib:
        raise ValueError("Required compiled C++ library not found.")
    return calcRXX_withSparseRespCSR_cpp(X, spR_csr)

def calcSpRXXT(X=None, spR_csr=None, **kwargs):
    ''' Compute expected outer-product statistic for each state.

    Returns
    -------
    S_xxT : 3D array, K x D x D
    '''
    if not hasCPPLib:
        raise ValueError("Required compiled C++ library not found.")
    return calcRXXT_withSparseRespCSR_cpp(X, spR_csr)

def calcRlogR_withDenseResp(R=None, **kwargs):
    return -1 * calcRlogR(R)

def calcRXX_withSparseResp_numpy_forloop(spR_csc=None, X=None, **kwargs):
    '''
    '''
    N, K = spR_csc.shape
    NX, D = X.shape
    assert N == NX

    stat_XX = np.zeros((K,D))
    for k in range(K):
        start_k = spR_csc.indptr[k]
        stop_k = spR_csc.indptr[k+1]
        rowids_k = spR_csc.indices[start_k:stop_k]
        sqX_k = X[rowids_k].copy()
        np.square(sqX_k, out=sqX_k)
        np.dot(spR_csc.data[start_k:stop_k], sqX_k, out=stat_XX[k])
    return stat_XX

def calcRXX_withSparseResp_numpy_dot(spR_csc=None, X=None, **kwargs):
    '''
    '''
    N, K = spR_csc.shape
    NX, D = X.shape
    assert N == NX
    stat_XX = spR_csc.T * np.square(X)
    return stat_XX

def calcRXX_withDenseResp(R=None, X=None, **kwargs):
    N, K = R.shape
    NX, D = X.shape
    assert N == NX
    stat_XX = np.dot(R.T, np.square(X))
    return stat_XX

def calcRXXT_withDenseResp(R=None, X=None, **kwargs):
    N, K = R.shape
    NX, D = X.shape
    assert N == NX
    stat_XXT = np.zeros((K, D, D))
    for k in range(K):
        sqrtRX_k = np.sqrt(R[:,k])[:,np.newaxis] * X
        stat_XXT[k] = dotATA(sqrtRX_k)
        #RX_k = R[:, k][:,np.newaxis] * X
        #stat_XXT[k] = np.dot(RX_k.T, X)
    return stat_XXT


def make_funcList(prefix='calcRXX_'):
    funcList = []
    for key, val in list(globals().items()):
        if key.startswith(prefix):
            funcList.append(val)
    return [f for f in sorted(funcList)]

def test_speed(X=None, R=None, nnzPerRow=2,
                           N=100, K=3, D=2,
                           funcList=None,
                           prefix='calcRXX_',
                           nRep=1, **kwargs):
    if funcList is None:
        funcList = make_funcList(prefix=prefix)
    kwargs = _make_kwarg_dict(
            X=X, R=R, nnzPerRow=nnzPerRow, N=N, K=K, D=D)
    assert 'X' in kwargs
    for func in funcList:
        do_timing_test_for_func(func, (), kwargs, nRep=nRep)

def do_timing_test_for_func(func, args, kwargs, nRep=1):
    times = list()
    for trial in range(nRep):
        tstart = time.time()
        func(*args, **kwargs)
        tstop = time.time()
        times.append(tstop - tstart)
    print(" AVG %.4f sec  MEDIAN %.4f sec | %s" % (
        np.mean(times), np.median(times), func.__name__))


def test_correctness(X=None, R=None, nnzPerRow=2,
                                         N=100, K=3, D=2,
                                         funcList=None,
                                         prefix='calcRXX_'):
    if funcList is None:
        funcList = make_funcList(prefix=prefix)
    kwargs = _make_kwarg_dict(
            X=X, R=R, nnzPerRow=nnzPerRow, N=N, K=K, D=D)
    for i in range(len(funcList)):
        for j in range(i+1, len(funcList)):
            func_i = funcList[i]
            func_j = funcList[j]
            ans_i = func_i(**kwargs)
            ans_j = func_j(**kwargs)

            if prefix.count('calcRlogR') and kwargs['nnzPerRow'] == 1:
                # SPARSE routine gives scalar 0.0
                # but DENSE routine gives vector of all zeros
                ans_i = as1D(toCArray(ans_i))
                ans_j = as1D(toCArray(ans_j))
                if ans_i.size < K:
                    ans_j = np.sum(ans_j)
                elif ans_j.size < K:
                    ans_i = np.sum(ans_i)
            assert np.allclose(ans_i, ans_j)
    print('  all pairs of funcs give same answer')

def _make_kwarg_dict(
                X=None, R=None, nnzPerRow=2,
                N=100, K=3, D=2):
    if X is None:
        X = np.random.randn(N, D)
    if R is None:
        R = np.random.rand(N, K)
        R *= R
        R /= R.sum(axis=1)[:,np.newaxis]
    # Sparsify R
    spR_csr = sparsifyResp(R, nnzPerRow)
    spR_csc = spR_csr.tocsc()
    R = spR_csc.toarray()
    np.maximum(R, 1e-100, out=R) # avoid NaN values
    return dict(X=X, R=R, spR_csc=spR_csc, spR_csr=spR_csr,
            nnzPerRow=nnzPerRow)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--D', type=int, default=64)
    parser.add_argument('--nnzPerRow', type=int, default=2)
    parser.add_argument('--nRep', type=int, default=1)
    parser.add_argument('--prefix', type=str, default='calcRXX_')
    args = parser.parse_args()

    print('TESTING FUNCTIONS NAMED ', args.prefix)
    for nnzPerRow in [1, 2, 4]:
        print('nnzPerRow=%d' % (nnzPerRow))
        test_correctness(N=1, K=10, D=2,
                nnzPerRow=nnzPerRow, prefix=args.prefix)
        test_correctness(N=33, K=10, D=2,
                nnzPerRow=nnzPerRow, prefix=args.prefix)
    test_speed(**args.__dict__)
