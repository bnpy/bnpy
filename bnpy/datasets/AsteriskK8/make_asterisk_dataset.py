'''
AsteriskK8.py

Simple toy dataset of 8 Gaussian components with full covariance.

Generated data form well-separated blobs arranged in "asterisk" shape
when plotted in 2D.
'''
from builtins import *
import scipy.linalg
import numpy as np
import pandas as pd
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data import XData


K = 8
D = 2

w = np.asarray([1., 2., 1., 2., 1., 2., 1., 2.])
w = w / w.sum()

# Place means evenly spaced around a circle
Rad = 1.0
ts = np.linspace(0, 2 * np.pi, K + 1)
ts = ts[:-1]
Mu = np.zeros((K, D))
Mu[:, 0] = np.cos(ts)
Mu[:, 1] = np.sin(ts)

# Create basic 2D cov matrix with major axis much longer than minor one
V = 1.0 / 16.0
SigmaBase = np.asarray([[V, 0], [0, V / 100.0]])

# Create several Sigmas by rotating this basic covariance matrix
Sigma = np.zeros((K, D, D))
for k in range(K):
    Sigma[k] = rotateCovMat(SigmaBase, k * np.pi / 4.0)

# Precompute cholesky decompositions
cholSigma = np.zeros(Sigma.shape)
for k in range(K):
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k])


def sample_data_from_comp(k, Nk, PRNG):
    return Mu[k, :] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk)).T


def make_X(seed=0, nObsTotal=0):
    PRNG = np.random.RandomState(seed)
    trueList = list()
    Npercomp = PRNG.multinomial(nObsTotal, w)
    X = list()
    for k in range(K):
        X.append(sample_data_from_comp(k, Npercomp[k], PRNG))
        trueList.append(k * np.ones(Npercomp[k]))
    X = np.vstack(X)
    TrueZ = np.hstack(trueList)
    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ

def plot_true_clusters():
    from bnpy.viz import GaussViz
    for k in range(K):
        c = k % len(GaussViz.Colors)
        GaussViz.plotGauss2DContour(Mu[k], Sigma[k], color=GaussViz.Colors[c])

# Main
###########################################################
if __name__ == "__main__":
    X, TrueZ = make_X(seed=0, nObsTotal=5000)
    np.savez('x_dataset.npz',
        **dict(X=X, TrueZ=TrueZ, name='AsteriskK8'))
    scipy.io.savemat('x_dataset.mat',
        dict(X=X, TrueZ=TrueZ, name='AsteriskK8'))
    x_names = ['x_%d' % s for s in np.arange(X.shape[1])]
    z_names = 'z_true'
    x_df = pd.DataFrame(
        np.hstack([X, TrueZ[:,np.newaxis]]),
        columns=np.hstack([x_names, z_names]))
    x_df.to_csv("x_dataset.csv", index=False)
