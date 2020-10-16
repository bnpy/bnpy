'''
DeadLeaves.py

'''
import scipy.linalg
import numpy as np

from bnpy.data import XData

# User-facing ###########################################################
# Accessors


def get_short_name():
    return 'DeadLeavesD%d' % (D)


def get_data_info():
    return 'Dead Leaves Data. K=%d. D=%d.' % (K, D)


def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    X, TrueZ = generateData(seed, nObsTotal)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data

# Create True Params
# for generating data


def makeTrueParams(Din):
    ''' Create mixture weights, means, covariance matrices for all components
    '''
    global K, D
    D = Din
    K = 8
    global w
    global Mu
    global Sigma
    global cholSigma

    w = np.ones(K)
    w = w / np.sum(w)
    Mu = np.zeros((K, D))

    Sigma = np.zeros((K, D, D))
    cholSigma = np.zeros(Sigma.shape)
    for k in range(K):
        Sigma[k] = makeImgPatchCovMatForComp(D, k)
        cholSigma[k] = scipy.linalg.cholesky(Sigma[k])


def makeImgPatchPrototype(D, compID):
    ''' Create image patch prototype for specific component
        Returns
        --------
        Xprototype : sqrt(D) x sqrt(D) matrix
    '''
    # Create a "prototype" image patch of PxP pixels
    P = np.sqrt(D)
    Xprototype = np.zeros((P, P))
    if compID % 4 == 0:
        Xprototype[:P / 2] = 1.0
        Xprototype = np.rot90(Xprototype, compID / 4)
    if compID % 4 == 1:
        Xprototype[np.tril_indices(P)] = 1
        Xprototype = np.rot90(Xprototype, (compID - 1) / 4)
    if compID % 4 == 2:
        Xprototype[np.tril_indices(P, 2)] = 1
        Xprototype = np.rot90(Xprototype, (compID - 2) / 4)
    if compID % 4 == 3:
        Xprototype[np.tril_indices(P, -2)] = 1
        Xprototype = np.rot90(Xprototype, (compID - 3) / 4)
    return Xprototype


def makeImgPatchCovMatForComp(D, compID, sig=0.005):
    ''' Create img patch covariance matrix for specific component

        Returns
        --------
        Sigma : D x D covariance matrix
    '''
    # Create a "prototype" image patch of sqrt(D)xsqrt(D) pixels
    Xprototype = makeImgPatchPrototype(D, compID)

    # Now generate N "observed" image patches,
    #  each a D-dimensional vector, slight perturbations of the prototype
    N = 100 * D
    PRNG = np.random.RandomState(compID)
    Xnoise = sig * PRNG.randn(N, D)
    Xsignal = np.tile(Xprototype.flatten(), (N, 1))
    Xsignal *= PRNG.randn(N, 1)
    X = Xnoise + Xsignal

    # Finally, measure the covariance of the observed image patches
    Sigma = np.cov(X.T)
    return Sigma + 1e-5 * np.eye(D)


# Generate data
###########################################################
def sample_data_from_comp(k, Nk):
    PRNG = np.random.RandomState(k)
    return Mu[k, :] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk)).T


def generateData(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    trueList = list()
    Npercomp = PRNG.multinomial(nObsTotal, w)
    X = list()
    for k in range(K):
        X.append(sample_data_from_comp(k, Npercomp[k]))
        trueList.append(k * np.ones(Npercomp[k]))
    X = np.vstack(X)
    TrueZ = np.hstack(trueList)
    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ


# Plots/Visualization
###########################################################
def plotImgPatchPrototypes(doShowNow=True):
    from matplotlib import pylab
    pylab.figure()
    for kk in range(K):
        pylab.subplot(2, 4, kk + 1)
        Xp = makeImgPatchPrototype(D, kk)
        pylab.imshow(Xp, interpolation='nearest')
    if doShowNow:
        pylab.show()


def plotTrueCovMats(doShowNow=True):
    from matplotlib import pylab
    pylab.figure()
    for kk in range(K):
        pylab.subplot(2, 4, kk + 1)
        pylab.imshow(Sigma[kk], interpolation='nearest')
    if doShowNow:
        pylab.show()
