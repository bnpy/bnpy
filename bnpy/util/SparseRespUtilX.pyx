"""
References
----------
Neal Hughes
Blog post on "Fast Python loops with Cython"
http://nealhughes.net/cython1/
"""

import numpy as np

def calcSpRData_cython(double[:,:] R, long[:,:] ColIDs, int nnzPerRow):
    """
    """
    cdef int N = R.shape[0]
    cdef int K = R.shape[1]
    cdef double rowsum = 0.0
    cdef int start = 0
    cdef int curk = 0
    cdef double[:] spR_data = np.zeros(N * nnzPerRow, dtype=np.float64)
    for n in range(N):
        start = n * nnzPerRow
        rowsum = 0.0
        for nzk in range(nnzPerRow):
            curk = ColIDs[n,nzk]
            rowsum += R[n, curk]
        for nzk in range(nnzPerRow):
            curk = ColIDs[n,nzk]
            spR_data[start+nzk] = R[n, curk] / rowsum
    return np.asarray(spR_data)
