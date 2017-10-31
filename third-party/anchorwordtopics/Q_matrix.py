
import numpy as np
import time
import scipy.sparse
import math
from  helper_functions import *

# Given a sparse CSC document matrix M (with floating point entries),
# comptues the word-word correlation matrix Q
def generate_Q_matrix(M, words_per_doc=None):
    
    simulation_start = time.time()
    
    vocabSize = M.shape[0]
    numdocs = M.shape[1]
    
    diag_M = np.zeros(vocabSize)

    for j in range(M.indptr.size - 1):
        
        # start and end indices for column j
        start = M.indptr[j]
        end = M.indptr[j + 1]
        
        wpd = np.sum(M.data[start:end])
        if words_per_doc != None and wpd != words_per_doc:
            print('Error: words per doc incorrect')
        
        row_indices = M.indices[start:end]
        
        diag_M[row_indices] += M.data[start:end]/(wpd*(wpd-1))
        M.data[start:end] = M.data[start:end]/math.sqrt(wpd*(wpd-1))
    
    
    Q = M*M.transpose()/numdocs
    Q = Q.todense()
    Q = np.array(Q, copy=False)

    diag_M = diag_M/numdocs
    Q = Q - np.diag(diag_M)
    
    #print 'Sum of entries in Q is ', np.sum(Q)
    #print 'Multiplying Q took ', str(time.time() - simulation_start), 'seconds'
    
    return Q
