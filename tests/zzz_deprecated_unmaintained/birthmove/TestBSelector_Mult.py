import numpy as np
import bnpy
import sys

from bnpy.util.RandUtil import rotateCovMat
from TestBSelector import main

def makeDataset(K=5, Nk=100, Nvec=None, Nd=100, **kwargs):
    K = int(K)
    Nd = int(Nd)
    if Nvec is None:
        Nk = int(Nk)
        Nvec = Nk * np.ones(K)
    elif isinstance(Nvec, str):
        Nvec = Nvec.split(',')
    Nvec = np.asarray(Nvec, dtype=np.int32)

    Z = [None for k in range(K)]
    word_id = np.zeros(0)
    word_ct = np.zeros(0)
    doc_range = np.zeros(1)
    
    D = 25
    for k in range(K):
        PRNG = np.random.RandomState(k)
        thr = PRNG.rand()
        mu_k = PRNG.gamma(0.5, size=D)
        mu_k /= mu_k.sum()

        Z[k] = k * np.ones(Nvec[k])
        for docID in range(Nvec[k]):
            X_d = PRNG.multinomial(PRNG.poisson(Nd), mu_k)
            word_id_d = np.flatnonzero(X_d)
            word_ct_d = X_d[word_id_d]
            word_id = np.hstack([word_id, word_id_d])
            word_ct = np.hstack([word_ct, word_ct_d])
            doc_range = np.hstack(
                [doc_range, doc_range[-1] + word_id_d.size])

    Z = np.hstack(Z)
    Data = bnpy.data.BagOfWordsData(
        doc_range=doc_range,
        word_id=word_id,
        word_count=word_ct,
        vocab_size=D,
        TrueParams=dict(Z=Z))
    return Data

def makeModel(Data, lam=0.1, **kwargs):
    PriorArgs = dict(lam=lam)
    PriorArgs.update(kwargs)
    allocModel = bnpy.allocmodel.DPMixtureModel(
        'VB', gamma0=1.0)
    obsModel = bnpy.obsmodel.MultObsModel(
        'VB', Data=Data, **PriorArgs)
    model = bnpy.HModel(allocModel, obsModel)
    return model


if __name__ == '__main__':
    main(makeDataset=makeDataset, makeModel=makeModel)
