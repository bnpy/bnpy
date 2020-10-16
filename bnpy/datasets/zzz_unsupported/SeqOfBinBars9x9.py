'''
SeqOfBinBars9x9.py

Binary toy bars data, with a 9x9 grid,
so each observation is a vector of size 81.

There are K=20 true topics
* one common background topic (with prob of 0.05 for all pixels)
* one rare foreground topic (with prob of 0.90 for all pixels)
* 18 bar topics, one for each row/col of the grid.

The basic idea is that the background topic is by far most common.
It takes over 50% of all timesteps.
The horizontal bars and the vertical bars form coherent groups,
where we transition between each bar (1-9) in a standard step-by-step way.

The rare foreground topic simulates the rare "artificial" phenomena
reported by some authors, of unusual all-marks-on bursts in chr data.
'''
import os
import sys
import scipy.io
import numpy as np
from bnpy.data import GroupXData
from bnpy.util import as1D

K = 20  # Number of topics
D = 81  # Vocabulary Size

bgStateID = 18
fgStateID = 19

Defaults = dict()
Defaults['nDocTotal'] = 50
Defaults['T'] = 10000
Defaults['bgProb'] = 0.05
Defaults['fgProb'] = 0.90
Defaults['seed'] = 8675309
Defaults['maxTConsec'] = Defaults['T'] / 5.0


def get_data(**kwargs):
    ''' Create dataset as bnpy DataObj object.
    '''
    Data = generateDataset(**kwargs)
    Data.name = 'SeqOfBinBars9x9'
    Data.summary = 'Binary Bar Sequences with %d true topics.' % (K)
    return Data


def makePi(stickyProb=0.95, extraStickyProb=0.9999,
           **kwargs):
    ''' Make phi matrix that defines probability of each pixel.
    '''
    pi = np.zeros((K, K))
    # Horizontal bars
    for k in range(9):
        pi[k, k] = stickyProb
        if k == 8:
            pi[k, bgStateID] = 1 - stickyProb
        else:
            pi[k, (k + 1) % 9] = 1 - stickyProb

    # Vertical bars
    for k in range(9, 18):
        pi[k, k] = stickyProb
        if k == 17:
            pi[k, bgStateID] = 1 - stickyProb
        else:
            pi[k, 9 + (k + 1) % 9] = 1 - stickyProb

    pi[bgStateID, :] = 0.0
    pi[bgStateID, bgStateID] = extraStickyProb
    pi[bgStateID, 0] = 5.0 / 12 * (1 - extraStickyProb)
    pi[bgStateID, 9] = 5.0 / 12 * (1 - extraStickyProb)
    pi[bgStateID, fgStateID] = 2.0 / 12 * (1 - extraStickyProb)

    mstickyProb = 0.5 * (stickyProb + extraStickyProb)
    pi[fgStateID, :] = 0.0
    pi[fgStateID, fgStateID] = mstickyProb
    pi[fgStateID, bgStateID] = 1 - mstickyProb
    assert np.allclose(1.0, np.sum(pi, 1))
    return pi


def makePhi(fgProb=0.75, bgProb=0.05, **kwargs):
    ''' Make phi matrix that defines probability of each pixel.
    '''
    phi = bgProb * np.ones((K, np.sqrt(D), np.sqrt(D)))
    for k in range(18):
        if k < 9:
            rowID = k
            # Horizontal bars
            phi[k, rowID, :] = fgProb
        else:
            colID = k - 9
            phi[k, :, colID] = fgProb
    phi[-2, :, :] = bgProb
    phi[-1, :, :] = fgProb
    phi = np.reshape(phi, (K, D))
    return phi


def generateDataset(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    phi = makePhi(**kwargs)
    transPi = makePi(**kwargs)
    PRNG = np.random.RandomState(kwargs['seed'])

    nSeq = kwargs['nDocTotal']
    T_in = kwargs['T']

    if isinstance(T_in, str):
        Tvals = [int(T) for T in T_in.split(',')]
    else:
        Tvals = [T_in]

    if len(Tvals) == 1:
        seqLens = Tvals[0] * np.ones(nSeq, dtype=np.int32)
    elif len(Tvals) < nSeq:
        seqLens = np.tile(Tvals, nSeq)[:nSeq]
    elif len(Tvals) >= nSeq:
        seqLens = np.asarray(Tvals, dtype=np.int32)[:nSeq]

    doc_range = np.hstack([0, np.cumsum(seqLens)])
    N = doc_range[-1]
    allX = np.zeros((N, D))
    allZ = np.zeros(N, dtype=np.int32)

    startStates = [bgStateID, fgStateID]
    states0toKm1 = np.arange(K)
    # Each iteration generates one time-series/sequence
    # with starting state deterministically rotating among all states
    for i in range(nSeq):
        start = doc_range[i]
        stop = doc_range[i + 1]

        T = stop - start
        Z = np.zeros(T, dtype=np.int32)
        X = np.zeros((T, D))
        nConsec = 0

        Z[0] = startStates[i % len(startStates)]
        X[0] = PRNG.rand(D) < phi[Z[0]]
        for t in range(1, T):
            if nConsec > kwargs['maxTConsec']:
                # Force transition if we've gone on too long
                transPi_t = transPi[Z[t - 1]].copy()
                transPi_t[Z[t - 1]] = 0
                transPi_t /= transPi_t.sum()
            else:
                transPi_t = transPi[Z[t - 1]]
            Z[t] = PRNG.choice(states0toKm1, p=transPi_t)
            X[t] = PRNG.rand(D) < phi[Z[t]]
            if Z[t] == Z[t - 1]:
                nConsec += 1
            else:
                nConsec = 0
        allZ[start:stop] = Z
        allX[start:stop] = X

    TrueParams = dict()
    TrueParams['beta'] = np.mean(transPi, axis=0)
    TrueParams['phi'] = phi
    TrueParams['Z'] = allZ
    TrueParams['K'] = K
    return GroupXData(allX, doc_range=doc_range, TrueParams=TrueParams)

DefaultOutputDir = os.path.join(
    os.environ['XHMMROOT'], 'datasets', 'SeqOfBinBars9x9')


def saveDatasetToDisk(outputdir=DefaultOutputDir):
    ''' Save dataset to disk for scalable experiments.
    '''
    Data = get_data()
    for k in range(K):
        print('N[%d] = %d' % (k, np.sum(Data.TrueParams['Z'] == k)))

    # Save it as batches
    nDocPerBatch = 2
    nBatch = Data.nDocTotal // nDocPerBatch

    for batchID in range(nBatch):
        mask = np.arange(batchID * nDocPerBatch, (batchID + 1) * nDocPerBatch)
        Dbatch = Data.select_subset_by_mask(mask, doTrackTruth=1)

        outmatpath = os.path.join(
            outputdir,
            'batches/batch%02d.mat' %
            (batchID))
        Dbatch.save_to_mat(outmatpath)
    with open(os.path.join(outputdir, 'batches/Info.conf'), 'w') as f:
        f.write('datasetName = SeqOfBinBars9x9\n')
        f.write('nBatchTotal = %d\n' % (nBatch))
        f.write('nDocTotal = %d\n' % (Data.nDocTotal))

    Dsmall = Data.select_subset_by_mask([0, 1], doTrackTruth=1)
    Dsmall.save_to_mat(os.path.join(outputdir, 'HMMdataset.mat'))


if __name__ == '__main__':
    import scipy.io
    import bnpy.viz.BernViz as BernViz
    # saveDatasetToDisk()
    # BernViz.plotCompsAsSquareImages(Data.TrueParams['phi'])

    Data = get_data(nDocTotal=2)

    pylab = BernViz.pylab
    pylab.subplots(nrows=1, ncols=Data.nDoc)
    for d in range(2):
        start = Data.doc_range[d]
        stop = Data.doc_range[d + 1]
        pylab.subplot(1, Data.nDoc, d + 1)
        Xim = Data.X[start:stop]
        pylab.imshow(Xim,
                     interpolation='nearest', cmap='bone',
                     aspect=Xim.shape[1] / float(Xim.shape[0]),
                     )
        pylab.ylim([np.minimum(stop - start, 5000), 0])
    pylab.show(block=True)
