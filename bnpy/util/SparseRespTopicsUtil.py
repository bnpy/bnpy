import numpy as np
import scipy.sparse
import timeit
import time
import sys

try:
    import bnpy.util.lib.sparseResp.LibSparseRespTopics
    hasLibReady = bnpy.util.lib.sparseResp.LibSparseRespTopics.hasLibReady
except ImportError:
    hasLibReady = False

def fillInDocTopicCountFromSparseResp(Data, LP):
    if hasattr(Data, 'word_count'):
        for d in range(Data.nDoc):
            start = Data.doc_range[d]
            stop = Data.doc_range[d+1]
            spR_d = LP['spR'][start:stop]
            wc_d = Data.word_count[start:stop]
            LP['DocTopicCount'][d] = wc_d * spR_d
    else:
        for d in range(Data.nDoc):
            start = Data.doc_range[d]
            stop = Data.doc_range[d+1]
            spR_d = LP['spR'][start:stop]
            LP['DocTopicCount'][d] = spR_d.sum(axis=0)
    return LP
