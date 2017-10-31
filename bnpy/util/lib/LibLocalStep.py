'''
LibLocalStep.py

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
cint = ctypes.c_int
carrd = ndpointer(ctypes.c_double)
carri = ndpointer(ctypes.c_int)

# Compiled library is in the lib/ directory
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libpath = os.path.join(libpath, 'lib')
doUseLib = True
try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, 'libLocalStep.so'))
    lib.CalcDocTopicCount.restype = None
    lib.CalcDocTopicCount.argtypes = [cint, cint, cint, cint,
                                      carri, carri,
                                      carrd, carrd, carrd, carrd, carrd]
except OSError:
    # No compiled C++ library exists
    doUseLib = False


def calcDocTopicCount(*args, **kwargs):
    '''
    Args
    -------
    activeDocs : 1D array of int, size A <= D
    docptr : 1D array of int, size D+1
    word_count : 1D array, size Ntoken
    Prior : 2D array, size D x K
    Lik : 2D array, size Ntoken x K

    Optional Args (preallocated for speed)
    -------
    sumR : 1D array, size Ntoken
    DocTopicCount : 2D array, size D x K

    Returns
    -------
    sumR : 1D array, size Ntoken
    DocTopicCount : 2D array, size D x K
    '''
    if 'methodLP' in kwargs and kwargs['methodLP'] == 'c' and doUseLib:
        return calcDocTopicCount_c(*args)

    else:
        return calcDocTopicCount_numpy(*args)


def calcDocTopicCount_numpy(
        activeDocs, docIndices, word_count,
        Prior, Lik,
        sumR=None, DocTopicCount=None):
    if sumR is None:
        sumR = np.zeros(N)
    if DocTopicCount is None:
        DocTopicCount = np.zeros((D, K), order='C')
    if np.isfortran(DocTopicCount):
        print('here!!!!')
        DocTopicCount = np.ascontiguousarray(DocTopicCount)

    for d in activeDocs:
        start = docIndices[d]
        stop = docIndices[d + 1]
        Lik_d = Lik[start:stop]

        np.dot(Lik_d, Prior[d], out=sumR[start:stop])

        np.dot(word_count[start:stop] / sumR[start:stop],
               Lik_d,
               out=DocTopicCount[d, :]
               )

    DocTopicCount[activeDocs] *= Prior[activeDocs]
    return sumR, DocTopicCount


def calcDocTopicCount_c(
        activeDocs, docIndices, word_count,
        Prior, Lik,
        sumR=None, DocTopicCount=None):
    if not doUseLib:
        raise NotImplementedError('Library not found. Please recompile.')
    A = activeDocs.size
    D, K = Prior.shape
    N, K2 = Lik.shape
    assert K == K2
    if sumR is None:
        sumR = np.zeros(N)
    if DocTopicCount is None:
        print('HERE!!')
        DocTopicCount = np.zeros((D, K), order='F')
    if not np.isfortran(DocTopicCount):
        raise NotImplementedError('NEED FORTRAN ORDER')
    lib.CalcDocTopicCount(N, D, K, A,
                          activeDocs, docIndices,
                          word_count, Prior, Lik,
                          sumR, DocTopicCount)
    return sumR, DocTopicCount


def unpack_args(args):
    a, d, w, P, L, R, C = args
    return dict(activeDocs=a,
                docIndices=d,
                wordcount=w,
                Prior=P, Lik=L, sumR=R, DocTopicCount=C)


def make_quick_args(D=5, N=85, K=4, A=4, order='F'):

    PRNG = np.random.RandomState(18)

    if A == D:
        activeDocs = np.arange(D, dtype=np.int32)
    else:
        activeDocs = PRNG.choice(list(range(D)), A, replace=False)
        activeDocs = np.asarray(np.sort(activeDocs), dtype=np.int32)

    if D == 1:
        docIndices = np.asarray([0, N], dtype=np.int32)
    else:
        docIndices = PRNG.choice(list(range(N)), D - 1, replace=False)
        docIndices = np.hstack([0, np.sort(docIndices), N])
        docIndices = np.asarray(docIndices, dtype=np.int32)

    word_count = PRNG.rand(N)

    Lik = np.asarray(PRNG.rand(N, K), order=order)
    Lik /= Lik.sum(axis=1)[:, np.newaxis]
    Prior = np.asarray(PRNG.rand(D, K), order=order)
    Prior /= Prior.sum(axis=1)[:, np.newaxis]

    sumR = np.zeros(N)
    DTMat = np.zeros((D, K), order=order)
    return (activeDocs, docIndices, word_count,
            Prior, Lik, sumR, DTMat)


def test_speed(D=4000, N=400, K=100, A=None, nTrial=5):
    import time
    if A is None:
        A = D / 2
    N = N * D

    args = make_quick_args(D=D, N=N, K=K, A=A, order='C')
    args2 = make_quick_args(D=D, N=N, K=K, A=A, order='F')

    stime = time.time()
    for _ in range(nTrial):
        R, DTM = calcDocTopicCount_numpy(*args)
    etime = time.time() - stime
    print('%.5f sec  numpy' % (etime))

    stime = time.time()
    for _ in range(nTrial):
        R2, DTM2 = calcDocTopicCount_c(*args2)
    etime = time.time() - stime
    print('%.5f sec  c' % (etime))

    print(np.allclose(R, R2))
    print(np.allclose(DTM, DTM2))


def test_quick_c():
    args = make_quick_args()
    Q = unpack_args(args)

    sR, DTM = calcDocTopicCount_c(*args)
    pprint_arr(sR, 'sumR')
    pprint_arr(DTM, 'DocTopicCount')


def test_quick_numpy():
    args = make_quick_args()
    Q = unpack_args(args)
    sR, DTM = calcDocTopicCount_numpy(*args)
    pprint_arr(sR, 'sumR')
    pprint_arr(DTM, 'DocTopicCount')


def pprint_arr(X, label, Kmax=10, fmt='%8.5f'):
    print('------------------', label)
    if X.ndim == 1:
        print(' '.join([fmt % x for x in X[:Kmax]]))
    else:
        nRow = np.minimum(Kmax, X.shape[0])
        for row in range(nRow):
            print(' '.join([fmt % x for x in X[row, :Kmax]]))


if __name__ == '__main__':

    print('============================ Numpy')
    test_quick_numpy()

    print('============================ C')
    test_quick_c()

    test_speed()
