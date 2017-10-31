from builtins import *
import numpy as np
import time
import os

hasCython = True
try:
    from EntropyUtilX import \
        calcRlogR_cython, calcRlogR_1D_cython, \
        calcRlogRdotv_cython, calcRlogRdotv_1D_cython
except ImportError:
    hasCython = False

hasNumexpr = True
try:
    import numexpr
except ImportError:
    hasNumexpr = False

# Configure num threads for numexpr
if hasNumexpr and 'OMP_NUM_THREADS' in os.environ:
    try:
        nThreads = int(os.environ['OMP_NUM_THREADS'])
        numexpr.set_num_threads(nThreads)
    except TypeError as ValueError:
        print('Unrecognized OMP_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
        pass

def calcRlogR(R, algVersion='cython'):
    ''' Compute R * log(R), then sum along columns of result.

    Returns
    --------
    H : 1D array, size K
        H[k] = sum(R[:,k] * log(R[:,k]))
    '''
    if hasCython and algVersion.count('cython'):
        if R.ndim == 1:
            return calcRlogR_1D_cython(R)
        else:
            return calcRlogR_cython(R)
    elif hasNumexpr and algVersion.count('numexpr'):
        return calcRlogR_numexpr(R)
    else:
        return calcRlogR_numpy_vectorized(R)

def calcRlogRdotv(R, v, algVersion='cython'):
    ''' Computes R * log(R) and takes inner product with weight vector v.

    Returns
    --------
    H : 1D array, size K
        H[k] = inner(v, R[:,k] * log(R[:,k]))
    '''
    if hasCython and algVersion.count('cython'):
        if R.ndim == 1:
            return calcRlogRdotv_1D_cython(R, v)
        else:
            return calcRlogRdotv_cython(R, v)
    elif hasNumexpr and algVersion.count('numexpr'):
        return calcRlogRdotv_numexpr(R, v)
    else:
        return calcRlogRdotv_numpy(R, v)

def test_correctness_calcRlogR(N=1000, K=10):
    R = np.random.rand(N, K)
    H1 = calcRlogR_numpy_forloop(R)
    H2 = calcRlogR_numpy_vectorized(R)
    H3 = calcRlogR_cython(R)
    H4 = calcRlogR_numexpr(R)
    assert np.allclose(H1, H2)
    assert np.allclose(H1, H3)
    assert np.allclose(H1, H4)

def test_correctness_calcRlogRdotv(N=1000, K=10):
    R = np.random.rand(N, K)
    v = np.random.rand(N)
    H1 = calcRlogRdotv_numpy(R, v)
    H2 = calcRlogRdotv_cython(R, v)
    H3 = calcRlogRdotv_numexpr(R, v)
    H4 = calcRlogRdotv(R, v)
    assert np.allclose(H1, H2)
    assert np.allclose(H1, H3)
    assert np.allclose(H1, H4)


def test_speed_calcRlogR(N=1000, K=10, nRep=1):
    R = np.random.rand(N, K)

    for calcRlogR_func in [
            calcRlogR_numpy_forloop,
            calcRlogR_numpy_vectorized,
            calcRlogR_cython,
            calcRlogR_numexpr]:
        if N * K > 5e5 and calcRlogR_func.__name__.count('forloop'):
            # For loops take too long on big problems
            print(" skipped %s" % (calcRlogR_func.__name__))
            continue
        do_timing_test_for_func(calcRlogR_func, (R,), nRep=nRep)

def do_timing_test_for_func(func, args, nRep=1):
    times = list()
    for trial in range(nRep):
        tstart = time.time()
        func(*args)
        tstop = time.time()
        times.append(tstop - tstart)
    print(" AVG %.4f sec  MEDIAN %.4f sec | %s" % (
        np.mean(times), np.median(times), func.__name__))

def test_speed_calcRlogRdotv(N=1000, K=10, nRep=1):
    R = np.random.rand(N, K)
    v = np.random.rand(N)
    for calcRlogRdotv_func in [
            calcRlogRdotv_numpy,
            calcRlogRdotv_cython,
            calcRlogRdotv_numexpr]:
        do_timing_test_for_func(calcRlogRdotv_func, (R,v,), nRep=nRep)

def calcRlogR_numexpr(R):
    """ Compute sum over columns of R * log(R). O(NK) memory. Vectorized.

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
    if R.shape[0] > 1:
        return numexpr.evaluate("sum(R*log(R), axis=0)")
    else:
        # Edge case: numexpr somehow fails if R has shape (1,K)
        return calcRlogR_numpy(R)

def calcRlogR_numpy_vectorized(R):
    """ Compute sum over columns of R * log(R). O(NK) memory. Vectorized.

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
    H = np.sum(R * np.log(R), axis=0)
    return H

def calcRlogR_numpy_forloop(R):
    """ Compute sum over columns of R * log(R). O(K) memory. Slow for loops.

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
    N = R.shape[0]
    K = R.shape[1]
    H = np.zeros(K)
    for n in range(N):
        for k in range(K):
            H[k] += R[n,k] * np.log(R[n,k])
    return H

def calcRlogRdotv_numpy(R, v):
    return np.dot(v, R * np.log(R))

def calcRlogRdotv_numexpr(R, v):
    RlogR = numexpr.evaluate("R*log(R)")
    return np.dot(v, RlogR)

def sumWithLotsOfMem(N, nRep=10):
    print(nRep)
    R = np.random.rand(N)
    time.sleep(0.5)
    for i in range(nRep):
        s = np.sum(R * R)
        time.sleep(0.1)
        del s
        time.sleep(0.1)
    return 0

def sumWithLotsOfMemMultiLine(N, nRep=10):
    print(nRep)
    R = np.random.rand(N)
    time.sleep(0.5)
    for i in range(nRep):
        Q = R * R
        time.sleep(0.1)
        s = np.sum(Q)
        time.sleep(0.1)
        del Q
        time.sleep(0.1)
    return 0

def sumWithNoMem(N, nRep=10):
    R = np.random.rand(N)
    for i in range(nRep):
        s = 0
        for i in range(N):
            s += R[i] * R[i]
    return s

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--nRep', type=int, default=1)
    parser.add_argument('--funcToEval', type=str, default='calcRlogR')
    parser.add_argument('--testType', type=str, default='speed')
    args = parser.parse_args()

    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'
    print('OMP_NUM_THREADS=', os.environ['OMP_NUM_THREADS'])
    print('N=', args.N)
    print('K=', args.K)

    if args.funcToEval.count('memtest'):
        from memory_profiler import memory_usage
        intervalSec = .0001
        taskTuple = (
            sumWithLotsOfMem,
            (args.N, args.nRep),)
        usageMiB = memory_usage(
            proc=taskTuple, interval=intervalSec, timeout=100)
        usageMiB = np.asarray(usageMiB)
        usageMiB -= usageMiB[0] # baseline = 0 at time 0
        elapsedTime = intervalSec * np.arange(len(usageMiB))

        from matplotlib import pylab
        pylab.plot(elapsedTime, usageMiB)
        pylab.xlabel('time (sec)')
        pylab.ylabel('memory (MiB)')
        pylab.show(block=True)

    elif args.funcToEval.count('dotv'):
        if args.testType.count('correctness'):
            test_correctness_calcRlogRdotv(args.N, args.K)
        else:
            test_speed_calcRlogRdotv(args.N, args.K, args.nRep)
    else:
        if args.testType.count('correctness'):
            test_correctness_calcRlogR(args.N, args.K)
        else:
            test_speed_calcRlogR(args.N, args.K, args.nRep)
