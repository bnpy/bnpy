import numpy as np
import bnpy
import sys

from bnpy.util.RandUtil import rotateCovMat
from TestBSelector import main

def makeDataset(K=5, Nk=100, Nvec=None, **kwargs):
    K = int(K)
    if Nvec is None:
        Nk = int(Nk)
        Nvec = Nk * np.ones(K)
    elif isinstance(Nvec, str):
        Nvec = Nvec.split(',')
    Nvec = np.asarray(Nvec, dtype=np.int32)

    Z = [None for k in range(K)]
    X = [None for k in range(K)]
    D = 25
    for k in range(K):
        PRNG = np.random.RandomState(k)
        thr = PRNG.rand()
        mu_k = 0.05 * np.ones(D)
        mu_k += 0.9 * (PRNG.rand(D) < thr)
        Z[k] = k * np.ones(Nvec[k])
        X[k] = PRNG.rand(Nvec[k], D) < mu_k[np.newaxis,:]
        
    Z = np.hstack(Z)
    X = np.vstack(X)
    return bnpy.data.XData(X=X, TrueZ=Z)


def makeModel(Data, lam0=0.1, lam1=0.1, **kwargs):
    PriorArgs = dict(lam1=lam1, lam0=lam0)
    PriorArgs.update(kwargs)
    allocModel = bnpy.allocmodel.DPMixtureModel(
        'VB', gamma0=1.0)
    obsModel = bnpy.obsmodel.BernObsModel(
        'VB', Data=Data, **PriorArgs)
    model = bnpy.HModel(allocModel, obsModel)
    return model


if __name__ == '__main__':
    main(makeDataset=makeDataset, makeModel=makeModel)
