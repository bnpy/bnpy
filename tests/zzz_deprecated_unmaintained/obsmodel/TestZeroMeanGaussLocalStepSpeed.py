import numpy as np
import scipy.linalg
import argparse
import time    
from contextlib import contextmanager


def measureTime(f, nTrial=3):
    def f_timer(*args, **kwargs):
        times = list()
        for rep in range(nTrial):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            times.append(end-start)
            if rep == 0:
                print("trial  %2d/%2d: %.3f sec %s" % (
                    rep+1, nTrial, times[-1], f.__name__))
            else:
                print("trial  %2d/%2d: %.3f sec" % (
                    rep+1, nTrial, times[-1]))
        print("mean   of %2d: %.3f sec" % (
            nTrial, np.mean(times)))
        print("median of %2d: %.3f sec" % (
            nTrial, np.median(times)))
        print('')
        return result
    return f_timer

@measureTime
def mahalDist_np_solve(X=None, B=None, cholB=None):
    ''' Compute mahalanobis the old fashioned way.
    '''
    if B is not None:
        cholB = np.linalg.cholesky(B)
    Q = np.linalg.solve(cholB, X.T)
    return Q

@measureTime
def mahalDist_scipy_solve(X=None, B=None, cholB=None):
    ''' Compute mahalanobis the old fashioned way.
    '''
    if B is not None:
        cholB = np.linalg.cholesky(B)
    Q = scipy.linalg.solve(cholB, X.T)
    return Q


@measureTime
def mahalDist_scipy_solve_triangular(X=None, B=None, cholB=None):
    ''' Compute mahalanobis with triangular method
    '''
    if B is not None:
        cholB = np.linalg.cholesky(B)
    Q = scipy.linalg.solve_triangular(cholB, X.T, lower=True)
    return Q

@measureTime
def mahalDist_scipy_solve_triangular_nocheck(
        X=None, B=None, cholB=None):
    ''' Compute mahalanobis with triangular method
    '''
    if B is not None:
        cholB = np.linalg.cholesky(B)
    Q = scipy.linalg.solve_triangular(
        cholB, X.T, lower=True, check_finite=False)
    return Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1e5)
    parser.add_argument('--D', type=int, default=64)
    args = parser.parse_args()
    N = args.N
    D = args.D

    print("TIMING TEST: N=%d D=%d" % (N, D))
    X = np.random.randn(N, D)
    R = np.random.randn(D, D)
    B = np.dot(R.T, R) + np.eye(D, D)
    cholB = np.linalg.cholesky(B)
    mahalDist_np_solve(X=X, cholB=cholB)
    mahalDist_scipy_solve(X=X, cholB=cholB)
    mahalDist_scipy_solve_triangular(X=X, cholB=cholB)
    mahalDist_scipy_solve_triangular_nocheck(X=X, cholB=cholB)
"""
In [41]: Qs = scipy.linalg.solve_triangular(cholB, X.T, lower=True, check_finite=False)

In [42]: %timeit -n1 -r1 Q = scipy.linalg.solve_triangular(cholB, X.T, lower=True, check_finite=False)
1 loops, best of 1: 625 ms per loop

In [43]: %timeit -n1 -r1 Q = scipy.linalg.solve_triangular(cholB, X.T, lower=True, check_finite=False)
1 loops, best of 1: 623 ms per loop

In [44]: %timeit -n1 -r1 Q = scipy.linalg.solve_triangular(cholB, X.T, lower=True)
1 loops, best of 1: 790 ms per loop

In [45]: %timeit -n1 -r1 Q = scipy.linalg.solve_triangular(cholB, X.T, lower=True)
1 loops, best of 1: 799 ms per loop

In [46]: %timeit -n1 -r1 Q = scipy.linalg.solve(cholB, X.T)
1 loops, best of 1: 1.26 s per loop

In [47]: %timeit -n1 -r1 Q = scipy.linalg.solve(cholB, X.T)
1 loops, best of 1: 1.26 s per loop

"""
