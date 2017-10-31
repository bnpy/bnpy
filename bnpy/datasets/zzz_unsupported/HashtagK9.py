'''
HashtagK9.py

Simple toy dataset of 9 Gaussian components with diagonal covariance structure.

Generated data form a "hashtag"-like shapes when plotted in 2D.
'''

import scipy.linalg
import numpy as np
from bnpy.data import XData


# User-facing fcns
###########################################################
def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    '''
      Args
      -------
      seed : integer seed for random number generator,
              used for actually *generating* the data
      nObsTotal : total number of observations for the dataset.

      Returns
      -------
        Data : bnpy XData object, with nObsTotal observations
    '''
    X, TrueZ = generate_data(seed, nObsTotal)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_data_info():
    return 'Hashtag Toy Data. Ktrue=%d. D=%d.' % (K, D)


def get_short_name():
    return 'HashtagK9'

# Create weights w
###########################################################
K = 9
D = 2

wExtra = 0.05
wH = 3. / 5 * (1.0 - wExtra)
wV = 2. / 5 * (1.0 - wExtra)
w = np.asarray([wH /
                4, wH /
                4, wH /
                4, wH /
                4, wV /
                4, wV /
                4, wV /
                4, wV /
                4, wExtra])

assert np.allclose(np.sum(w), 1.0)

# Create means Mu
###########################################################
Mu = np.asarray(
    [[-4, -1],
     [-4, +1],
     [4, -1],
     [4, +1],
     [-5, 0],
     [-3, 0],
     [3, 0],
     [5, 0],
     [0, 0],
     ], dtype=np.float64)
# Shift left-side down, right-side up
#  to break symmetry
Mu[Mu[:, 0] > 0, 1] += 0.5
Mu[Mu[:, 0] < 0, 1] -= 0.5

# Create covars Sigma
###########################################################
Sigma = np.zeros((K, D, D))
cholSigma = np.zeros((K, D, D))
Vmajor = 2.0
Vminor = Vmajor / 100
SigmaHoriz = np.asarray(
    [[Vmajor, 0],
     [0, Vminor]
     ])
SigmaVert = np.asarray(
    [[Vminor, 0],
     [0, Vmajor]
     ])
SigmaExtra = np.asarray(
    [[25 * Vmajor, 0],
     [0, Vmajor]
     ])

for k in range(K):
    if k < 4:
        Sigma[k] = SigmaHoriz
    elif k < 8:
        Sigma[k] = SigmaVert
    else:
        Sigma[k] = SigmaExtra
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k], lower=True)

# Generate Raw Data
###########################################################


def generate_data(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    trueList = list()
    Npercomp = PRNG.multinomial(nObsTotal, w)
    X = list()
    for k in range(K):
        X.append(sample_data_from_comp(k, Npercomp[k], PRNG))
        trueList.append(k * np.ones(Npercomp[k]))
    X = np.vstack(X)
    TrueZ = np.hstack(trueList)
    # Shuffle the ordering of observations,
    # so we don't have all examples from comp1 followed by all examples from
    # comp2
    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ


def sample_data_from_comp(k, Nk, PRNG):
    return Mu[k, :] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk)).T


# Visualize clusters
###########################################################
def plot_true_clusters():
    from bnpy.viz import GaussViz
    for k in range(K):
        c = k % len(GaussViz.Colors)
        GaussViz.plotGauss2DContour(Mu[k], Sigma[k], color=GaussViz.Colors[c])

# Main
if __name__ == "__main__":
    from matplotlib import pylab
    pylab.figure()
    X, TrueZ = generate_data(42, 10000)
    pylab.plot(X[:, 0], X[:, 1], 'k.')
    plot_true_clusters()
    pylab.axis('image')
    pylab.show(block=True)
