'''
AdmixAsteriskK8.py

Toy dataset of 8 Gaussian components with full covariance.

Generated data form well-separated blobs arranged in "asterisk" shape
when plotted in 2D.
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data import GroupXData

# User-facing


def get_data(seed=8675309, nDocTotal=2000, nObsPerDoc=100, **kwargs):
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
    Data = MakeGroupData(seed, nDocTotal, nObsPerDoc)
    Data.name = 'AdmixAsteriskK8'
    Data.summary = get_data_info()
    return Data


def get_short_name():
    ''' Return short string used in filepaths to store solutions
    '''
    return 'AdmixAsteriskK8'


def get_data_info():
    return 'Admixture Asterisk Toy Data. %d true clusters.' % (K)


def showExampleDocs(pylab=None, nrows=3, ncols=3):
    if pylab is None:
        from matplotlib import pylab
    Data = get_data(seed=0, nObsPerDoc=200)
    PRNG = np.random.RandomState(0)
    chosenDocs = PRNG.choice(Data.nDoc, nrows * ncols, replace=False)
    for ii, d in enumerate(chosenDocs):
        start = Data.doc_range[d]
        stop = Data.doc_range[d + 1]
        Xd = Data.X[start:stop]
        pylab.subplot(nrows, ncols, ii + 1)
        pylab.plot(Xd[:, 0], Xd[:, 1], 'k.')
        pylab.axis('image')
        pylab.xlim([-1.5, 1.5])
        pylab.ylim([-1.5, 1.5])
        pylab.xticks([])
        pylab.yticks([])
    pylab.tight_layout()
# Set Toy Parameters
###########################################################

K = 8
D = 2

gamma = 0.2

# Create "true" mean parameters
# Placed evenly spaced around a circle
Rad = 1.0
ts = np.linspace(0, 2 * np.pi, K + 1)
ts = ts[:-1]
Mu = np.zeros((K, D))
Mu[:, 0] = np.cos(ts)
Mu[:, 1] = np.sin(ts)

# Create "true" covariance parameters
# Each is a rotation of a template with major axis much larger than minor one
V = 1.0 / 16.0
SigmaBase = np.asarray([[V, 0], [0, V / 100.0]])
Sigma = np.zeros((K, D, D))
for k in range(K):
    Sigma[k] = rotateCovMat(SigmaBase, k * np.pi / 4.0)
# Precompute cholesky decompositions
cholSigma = np.zeros(Sigma.shape)
for k in range(K):
    cholSigma[k] = scipy.linalg.cholesky(Sigma[k])


def MakeGroupData(seed, nDoc, nObsPerDoc):
    ''' Make a GroupXData object
    '''
    PRNG = np.random.RandomState(seed)
    Pi = PRNG.dirichlet(gamma * np.ones(K), size=nDoc)
    XList = list()
    ZList = list()
    for d in range(nDoc):
        Npercomp = PRNG.multinomial(nObsPerDoc, Pi[d])
        for k in range(K):
            if Npercomp[k] < 1:
                continue
            Xcur_k = _sample_data_from_comp(k, Npercomp[k], PRNG)
            XList.append(Xcur_k)
            ZList.append(k * np.ones(Npercomp[k]))

    doc_range = np.arange(0, nDoc * nObsPerDoc + 1, nObsPerDoc)
    X = np.vstack(XList)
    TrueZ = np.hstack(ZList)
    return GroupXData(X, doc_range, TrueZ=TrueZ)


def _sample_data_from_comp(k, Nk, PRNG):
    return Mu[k, :] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk)).T

# Main
###########################################################


def plot_true_clusters():
    from bnpy.viz import GaussViz
    for k in range(K):
        c = k % len(GaussViz.Colors)
        GaussViz.plotGauss2DContour(Mu[k], Sigma[k], color=GaussViz.Colors[c])

if __name__ == "__main__":
    from matplotlib import pylab
    pylab.figure()
    plot_true_clusters()
    pylab.show(block=True)
