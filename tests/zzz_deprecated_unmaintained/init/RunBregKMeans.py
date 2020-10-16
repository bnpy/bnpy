'''
RunBregKMeans : verify monotonic improvement in k-means objective

Usage
-----
$ python RunBregKMeans.py <obsModelName> --N 100 --K 33
'''



import numpy as np
import argparse
import bnpy
runBregKMeans = bnpy.init.FromScratchBregman.runKMeans_BregmanDiv

from bnpy.viz.PlotUtil import pylab

def test_DiagGauss(K=50, N=1000, D=1, W=None, eps=1e-10, nu=0.001, kappa=0.001):
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=N)
    if D < Data.X.shape[1]:
        Data = bnpy.data.XData(X=Data.X[:,:D])
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'DiagGauss',
        dict(gamma0=10),
        dict(ECovMat='eye', sF=0.5, nu=nu, kappa=kappa),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N: 
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    else:
        W = None
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0, smoothFracInit=1.0,
        logFunc=print, eps=eps)
    try:
        assert np.all(np.diff(Lscores) <= 0)
    except AssertionError:
        from IPython import embed; embed()
    return Z, Mu, Lscores


def test_Gauss(K=50, N=1000, D=1, W=None, eps=1e-10, nu=0.001, kappa=0.001):
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=N)
    if D < Data.X.shape[1]:
        Data = bnpy.data.XData(X=Data.X[:,:D])
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Gauss',
        dict(gamma0=10),
        dict(ECovMat='eye', sF=0.5, nu=nu, kappa=kappa),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N: 
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    else:
        W = None
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0, smoothFracInit=1.0,
        logFunc=print, eps=eps)
    try:
        assert np.all(np.diff(Lscores) <= 0)
    except AssertionError:
        from IPython import embed; embed()
    return Z, Mu, Lscores

def test_ZeroMeanGauss(K=50, N=1000, D=1, W=None, eps=1e-10):
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=N)
    if D < Data.X.shape[1]:
        Data = bnpy.data.XData(X=Data.X[:,:D])
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'ZeroMeanGauss',
        dict(gamma0=10),
        dict(ECovMat='eye', sF=0.5, nu=0.01),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N: 
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    else:
        W = None
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0, smoothFracInit=1.0,
        logFunc=print, eps=eps)
    assert np.all(np.diff(Lscores) <= 0)
    return Z, Mu, Lscores

def test_Bern(K=50, N=1000, W=None, **kwargs):
    import SeqOfBinBars9x9
    Data = SeqOfBinBars9x9.get_data(nDocTotal=N, T=1)
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Bern',
        dict(gamma0=10),
        dict(lam1=0.1, lam0=0.1),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N:   
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    else:
        W = None
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0.0, smoothFracInit=1.0,
        logFunc=print)
    assert np.all(np.diff(Lscores) <= 0)

def test_Mult(K=50, N=1000, W=None, **kwargs):
    import BarsK10V900
    Data = BarsK10V900.get_data(nWordsPerDoc=33, nDocTotal=N)
    X = Data.getDocTypeCountMatrix()
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Mult',
        dict(gamma0=10),
        dict(lam=0.01),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N:   
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    else:
        W = None
    Z, Mu, Lscores = runBregKMeans(
        X, K, hmodel.obsModel,
        W=W, smoothFrac=0.0, smoothFracInit=1.0,
        logFunc=print)
    assert np.all(np.diff(Lscores) <= 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('obsModelName', type=str, default='Bern')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--D', type=int, default=2)
    parser.add_argument('--W', type=int, default=0)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument('--Nrange', type=str,
        default='5,10,33,211,345,500,1000')
    parser.add_argument('--Krange', type=str,
        default='3,5,8,10')
    args = parser.parse_args()
    if args.N is not None:
        Nrange = [args.N]
    else:
        Nrange = [int(n) for n in args.Nrange.split(',')]
    if args.K is not None:
        Krange = [args.K]
    else:
        Krange = [int(n) for n in args.Krange.split(',')]
    D = args.D
    W = args.W
    eps = args.eps

    for N in Nrange:
        for K in Krange:
            if K > N:
                continue
            print('--------- N=%d K=%d' % (N,K))
            if args.obsModelName.count("Bern"):
                test_Bern(K, N, W=W)
            elif args.obsModelName.count("Mult"):
                test_Mult(K, N, W=W)
            elif args.obsModelName.count("ZeroMean"):
                test_ZeroMeanGauss(K, N, D=D, W=W, eps=eps)
            elif args.obsModelName.count("Diag"):
                test_DiagGauss(K, N, D=D, W=W, eps=eps)
            else:
                test_Gauss(K, N, D=D, W=W, eps=eps)
