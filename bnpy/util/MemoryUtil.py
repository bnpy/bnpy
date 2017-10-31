from builtins import *
import os
import numpy as np
import scipy.sparse
import psutil

def getMemUsageOfCurProcess_MiB(field='rss'):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = getattr(process.memory_info_ex(), field)
    mem_MiB = mem / float(2 ** 20)
    return mem_MiB

def calcObjSize_MiB(arr):
    if hasattr(arr, "__dict__"):
        arr = arr.__dict__
    MiB_PER_BYTE = 1.0 / float(2**20)
    if isinstance(arr, np.ndarray):
        return arr.nbytes * MiB_PER_BYTE
    elif isinstance(arr, scipy.sparse.csr_matrix):
        nbyt = arr.data.nbytes + arr.indices.nbytes + arr.indptr.nbytes
        return nbyt * MiB_PER_BYTE
    elif isinstance(arr, dict):
        total = 0
        for key in arr:
            total += calcObjSize_MiB(arr[key])
        return total
    else:
        return 0
