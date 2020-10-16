'''
ToyHMMK4.py

Multiple sequences of data generated from a 4-state HMM.
Emissions come from 4 well-separated 2D Gaussians with no correlations.

Usage
-------
Data = ToyHMMK4.get_data()

Visualization
-------
From command-line,
>> python ToyHMMK4.py
will plot the true clusters, and color-segmented data for several sequences.

'''

import numpy as np
from bnpy.data import GroupXData
from bnpy.viz import GaussViz

import scipy.io


def get_data(seed=86758, seqLens=((3000, 3000, 3000, 3000, 500)), **kwargs):
    ''' Generate several data sequences, returned as a bnpy data-object

    Args
    -------
    seed : integer seed for random number generator,
          used for actually *generating* the data
    nObsTotal : total number of observations for the dataset.

    Returns
    -------
    Data : bnpy GroupXData object, with nObsTotal observations
    '''
    fullX, fullZ, seqIndicies = get_X(seed, seqLens)
    X = np.vstack(fullX)
    Z = np.asarray(fullZ)
    doc_range = np.asarray(seqIndicies)

    Data = GroupXData(X=X, doc_range=doc_range,
                      TrueZ=Z)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_short_name():
    return 'ToyHMMK4'


def get_data_info():
    return 'Toy data sequences, each using %d shared clusters.' % (K)

K = 4
D = 2

transPi = np.asarray([[0.9, 0.02, 0.04, 0.04],
                      [0.04, 0.9, 0.02, 0.04],
                      [0.04, 0.04, 0.9, 0.04],
                      [0.02, 0.04, 0.04, 0.9]])
initState = 1

mus = np.asarray([[0, 0],
                  [10, 0],
                  [0, 10],
                  [10, 10]])

sigmas = np.empty((K, D, D))
sigmas[0, :, :] = np.asarray([[2, 0], [0, 2]])
sigmas[1, :, :] = np.asarray([[2, 0], [0, 2]])
sigmas[2, :, :] = np.asarray([[2, 0], [0, 2]])
sigmas[3, :, :] = np.asarray([[2, 0], [0, 2]])


def get_X(seed, seqLens):
    ''' TODO
    '''
    prng = np.random.RandomState(seed)

    fullX = list()
    seqIndicies = list([0])
    fullZ = list()
    seqLens = list(seqLens)

    if len(np.shape(seqLens)) == 0:
        rang = range(1)
    else:
        rang = range(len(seqLens))

    # Each iteration generates one sequence
    for i in rang:
        Z = list()
        X = list()
        initX = prng.multivariate_normal(mus[initState, :],
                                         sigmas[initState, :, :])
        Z.append(initState)
        X.append(initX)
        for j in range(seqLens[i] - 1):
            trans = prng.multinomial(1, transPi[Z[j]])
            nextState = np.nonzero(trans)[0][0]
            nextX = prng.multivariate_normal(mus[nextState, :],
                                             sigmas[nextState, :, :])
            Z.append(nextState)
            X.append(nextX)

        fullZ = np.append(fullZ, Z)
        fullX.append(X)

        seqIndicies.append(seqLens[i] + seqIndicies[i])

    return (np.vstack(fullX),
            np.asarray(fullZ, dtype=np.int32),
            np.asarray(seqIndicies),
            )


# Main visualization
###########################################################
def plot_true_clusters():
    for k in range(K):
        c = k % len(GaussViz.Colors)
        GaussViz.plotGauss2DContour(
            mus[k],
            sigmas[k],
            color=GaussViz.Colors[c])


def plot_sequence(seqID, Data, dimID=0, maxT=200):
    Xseq = Data.X[Data.doc_range[seqID]:Data.doc_range[seqID + 1]]
    Zseq = Data.TrueParams['Z'][
        Data.doc_range[seqID]:Data.doc_range[
            seqID +
            1]]

    Xseq = Xseq[:maxT, dimID]  # Xseq is 1D after this statement!
    Zseq = Zseq[:maxT]

    # Plot X, colored by segments Z
    changePts = np.flatnonzero(np.abs(np.diff(Zseq)))
    changePts = np.hstack([0, changePts + 1])
    for ii, loc in enumerate(changePts[:-1]):
        nextloc = changePts[ii + 1]
        ts = np.arange(loc, nextloc)
        xseg = Xseq[loc:nextloc]
        kseg = int(Zseq[loc])

        color = GaussViz.Colors[kseg % len(GaussViz.Colors)]
        pylab.plot(ts, xseg, '.-', color=color, markersize=8)
        pylab.plot(
            [nextloc - 1, nextloc], [Xseq[nextloc - 1], Xseq[nextloc]], 'k:')
    pylab.ylim([-2, 14])

if __name__ == "__main__":
    from matplotlib import pylab
    pylab.figure()
    plot_true_clusters()
    Data = get_data(nObsTotal=5000)

    W = 10
    H = 15
    pylab.subplots(nrows=Data.nDoc, ncols=1, figsize=(W, H))
    for n in range(Data.nDoc):
        pylab.subplot(Data.nDoc, 1, n + 1)
        plot_sequence(n, Data)
    pylab.show(block=True)
