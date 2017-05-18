#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from scipy.linalg.cython_blas cimport dgemv
import numpy as np

from libc.math cimport exp

def calcRespInner_cython(double[:, :] resp, double[:] Zbar, double[:] wc_d, double[:, :] E_outer, double l_div_Nd_2):
    cdef int N = resp.shape[0]
    cdef int K = resp.shape[1]

    cdef double update = 0.0
    cdef double respsum = 0.0

    #Loop over tokens
    for i in range(N):
        
        #Subtract current token from Zbar
        for k in range(K):
            Zbar[k] -= wc_d[i] * resp[i, k]

        #Compute the update to the resp for token i
        respsum = 0.0
        for k in range(K):
            update = E_outer[k, k] 
            for k2 in range(K):
                update += 2 * Zbar[k2] * E_outer[k2, k]

            resp[i, k] *= exp(- l_div_Nd_2 * update)
            respsum += resp[i, k]

        #Normalize & update Zbar with the new resp
        for k in range(K):
            resp[i, k] /= respsum
            Zbar[k] += wc_d[i] * resp[i, k]


def calcRespInner_cython_blas(double[:, :] resp, double[:] Zbar, double[:] wc_d, double[:, :] E_outer, double l_div_Nd_2):
    cdef int N = resp.shape[0]
    cdef int K = resp.shape[1]


    cdef double[:] E_outer_d = np.zeros(K)
    for k in range(K):
        E_outer_d[k] = E_outer[k, k] 

    cdef double[:] update = np.zeros(K)
    cdef double respsum = 0.0

    cdef int one = 1
    cdef int two = 2
    cdef double done = - l_div_Nd_2 * 1.0
    cdef double dtwo = - l_div_Nd_2 * 2.0
    cdef char n = 'n'

    #Loop over tokens
    for i in range(N):
        
        #Subtract current token from Zbar
        for k in range(K):
            Zbar[k] -= wc_d[i] * resp[i, k]

        #Compute the update to the resp for token i
        respsum = 0.0
        for k in range(K):
            update[k] = E_outer_d[k] 

        dgemv(&n, &K, &K, &dtwo, &E_outer[0,0], &K, &Zbar[0], &one, &done, &update[0], &one)

        for k in range(K):
            resp[i, k] *= exp(update[k])
            respsum += resp[i, k]

        #Normalize & update Zbar with the new resp
        for k in range(K):
            resp[i, k] /= respsum
            Zbar[k] += wc_d[i] * resp[i, k]
   

def normalizeRows_cython(double[:, :] resp):
    cdef int N = resp.shape[0]
    cdef int K = resp.shape[1]
    cdef double respsum = 0.0

    for i in range(N):
        respsum = 0.0
        for k in range(K):
            respsum += resp[i, k]

        for k in range(K):
            resp[i, k] /= respsum
