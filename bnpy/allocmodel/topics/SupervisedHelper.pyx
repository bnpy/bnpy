#cython: boundscheck=False, wraparound=False, nonecheck=False

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
