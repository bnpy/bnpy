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
    for k in range(K):
        PRNG = np.random.RandomState(k)
        mu_k = 10 * PRNG.randn(2)
        Sigma_k = np.diag(PRNG.gamma(5, size=2)/5.0)
        Sigma_k = rotateCovMat(Sigma_k, np.pi * PRNG.rand())
        Z[k] = k * np.ones(Nvec[k])
        X[k] = PRNG.multivariate_normal(mu_k, Sigma_k, size=Nvec[k])
        
    Z = np.hstack(Z)
    X = np.vstack(X)
    return bnpy.data.XData(X=X, TrueZ=Z)


def makeModel(Data, ECovMat='diagcovdata', sF=1.0, **kwargs):
    PriorArgs = dict(sF=sF, ECovMat=ECovMat)
    PriorArgs.update(kwargs)
    allocModel = bnpy.allocmodel.DPMixtureModel(
        'VB', gamma0=1.0)
    obsModel = bnpy.obsmodel.GaussObsModel(
        'VB', Data=Data, **PriorArgs)
    model = bnpy.HModel(allocModel, obsModel)
    return model


if __name__ == '__main__':
    main(makeDataset=makeDataset, makeModel=makeModel)
