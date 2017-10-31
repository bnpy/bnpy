from builtins import *
import numpy as np
import os

from bnpy.data import BagOfWordsData
from TextFileReaderX import read_from_ldac_file

def LoadBagOfWordsDataFromFile_ldac_cython(filepath, nTokensPerByte=0.2, **kwargs):
    filesize_bytes = os.path.getsize(filepath)
    nUniqueTokens = int(nTokensPerByte *  filesize_bytes)
    try:
        dptr = np.zeros(nUniqueTokens, dtype=np.int32)
        wids = np.zeros(nUniqueTokens, dtype=np.int32)
        wcts = np.zeros(nUniqueTokens, dtype=np.float64)
        stop, dstop = read_from_ldac_file(
            filepath, nUniqueTokens, dptr, wids, wcts)
        return BagOfWordsData(
            word_id=wids[:stop],
            word_count=wcts[:stop],
            doc_range=dptr[:dstop],
            **kwargs)
    except IndexError as e:
        return LoadBagOfWordsDataFromFile_ldac(filepath, nTokensPerByte*2, **kwargs)

if __name__ == '__main__':
    #fpath = '/ltmp/testNYT.ldac'
    fpath = '/data/liv/textdatasets/nytimes/batches/batch111.ldac'
    vocab_size=8000

    fastD =  LoadBagOfWordsDataFromFile_ldac_cython(fpath, vocab_size=vocab_size)
    slowD = BagOfWordsData.LoadFromFile_ldac_python(fpath, vocab_size=vocab_size)
    print(fastD.word_id[:10])
    print(slowD.word_id[:10])
    assert np.allclose(fastD.word_id, slowD.word_id)
    assert np.allclose(fastD.word_count, slowD.word_count)
    assert np.allclose(fastD.doc_range, slowD.doc_range)
