'''
XViterbi.py

Learning alg extension for monitoring progress of HMM models.

This learning algorithm extension computes and stores optimal state sequences
for every time-series in the provided dataset. Saving will occur every
time a parameter-saving checkpoint is reached in the algorithm
(specified by the --saveEvery keyword argument).

Usage
--------
Add the following keyword arg to any call to bnpy.run
 --customFuncPath /path/to/bnpyrepo/bnpy/learnalg/extras/XViterbi.py

Example: use with MoCap6 dataset
$ python -m bnpy.Run MoCap6 FiniteHMM Gauss VB --K 10 --nLap 50 \
    --saveEvery 10
    --customFuncPath /path/to/bnpyrepo/bnpy/learnalg/extras/XViterbi.py

Notes
--------
Uses the custom-function interface for learning algorithms.
This interface means that the functions onAlgorithmComplete and onBatchComplete
defined here will be called at appropriate time in *every* learning algorithm.
See LearnAlg.py's eval_custom_function for details.
'''
import os
import numpy as np
import scipy.io

from bnpy.allocmodel.hmm.HMMUtil import runViterbiAlg
from bnpy.ioutil.ModelWriter import makePrefixForLap
from bnpy.util import StateSeqUtil

SavedLapSet = set()


def onAlgorithmComplete(**kwargs):
    ''' Runs viterbi at completion of the learning algorithm.

    Keyword Args
    --------
    All workspace variables passed along from learning alg.
    '''
    if kwargs['lapFrac'] not in SavedLapSet:
        runViterbiAndSave(**kwargs)


def onBatchComplete(**kwargs):
    ''' Runs viterbi whenever a parameter-saving checkpoint is reached.

    Keyword Args
    --------
    All workspace variables passed along from learning alg.
    '''
    global SavedLapSet
    if kwargs['isInitial']:
        SavedLapSet = set()

    isSaveParamsCheckpoint = kwargs['learnAlg'].isSaveParamsCheckpoint
    if not isSaveParamsCheckpoint(kwargs['lapFrac'], kwargs['iterid']):
        return
    if kwargs['lapFrac'] in SavedLapSet:
        return
    SavedLapSet.add(kwargs['lapFrac'])
    runViterbiAndSave(**kwargs)


def runViterbiAndSave(**kwargs):
    ''' Run viterbi alg on each sequence in dataset, and save to file.

    Keyword Args (all workspace variables passed along from learning alg)
    -------
    hmodel : current HModel object
    Data : current Data object
        representing *entire* dataset (not just one chunk)

    Returns
    -------
    None. MAP state sequences are saved to a MAT file.

    Output
    -------
    MATfile format: Lap0020.000MAPStateSeqs.mat
    '''
    if 'Data' in kwargs:
        Data = kwargs['Data']
    elif 'DataIterator' in kwargs:
        try:
            Data = kwargs['DataIterator'].Data
        except AttributeError:
            from bnpy.data.DataIteratorFromDisk import loadDataForSlice
            Dinfo = dict()
            Dinfo.update(kwargs['DataIterator'].DataInfo)
            if 'evalDataPath' in Dinfo:
                Dinfo['filepath'] = os.path.expandvars(Dinfo['evalDataPath'])
                Data = loadDataForSlice(**Dinfo)
            else:
                raise ValueError('DataIterator has no attribute Data')
    else:
        return None

    hmodel = kwargs['hmodel']
    lapFrac = kwargs['lapFrac']

    if 'savedir' in kwargs:
        savedir = kwargs['savedir']
    elif 'learnAlg' in kwargs:
        learnAlgObj = kwargs['learnAlg']
        savedir = learnAlgObj.savedir
        if hasattr(learnAlgObj, 'start_time'):
            elapsedTime = learnAlgObj.get_elapsed_time()
        else:
            elapsedTime = 0.0

    timestxtpath = os.path.join(savedir, 'times-saved-params.txt')
    with open(timestxtpath, 'a') as f:
        f.write('%.3f\n' % (elapsedTime))

    initPi = hmodel.allocModel.get_init_prob_vector()
    transPi = hmodel.allocModel.get_trans_prob_matrix()

    LP = hmodel.obsModel.calc_local_params(Data)
    Lik = LP['E_log_soft_ev']

    # Loop over each sequence in the collection
    zHatBySeq = list()
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        zHat = runViterbiAlg(Lik[start:stop], initPi, transPi)
        zHatBySeq.append(zHat)

    # Store MAP sequence to file
    prefix = makePrefixForLap(lapFrac)
    matfilepath = os.path.join(
        savedir,
        prefix +
        'MAPStateSeqs.mat')
    MATVarsDict = dict(
        zHatBySeq=StateSeqUtil.convertStateSeq_list2MAT(zHatBySeq))
    scipy.io.savemat(matfilepath, MATVarsDict, oned_as='row')

    zHatFlat = StateSeqUtil.convertStateSeq_list2flat(zHatBySeq, Data)
    Keff = np.unique(zHatFlat).size
    Kefftxtpath = os.path.join(savedir, 'Keff-saved-params.txt')
    with open(Kefftxtpath, 'a') as f:
        f.write('%d\n' % (Keff))

    Ktotal = hmodel.obsModel.K
    Ktotaltxtpath = os.path.join(savedir, 'Ktotal-saved-params.txt')
    with open(Ktotaltxtpath, 'a') as f:
        f.write('%d\n' % (Keff))

    # Save sequence aligned to truth and calculate Hamming distance
    if (hasattr(Data, 'TrueParams')) and ('Z' in Data.TrueParams):
        zHatFlatAligned = StateSeqUtil.alignEstimatedStateSeqToTruth(
            zHatFlat,
            Data.TrueParams['Z'])

        zHatBySeqAligned = StateSeqUtil.convertStateSeq_flat2list(
            zHatFlatAligned, Data)
        zHatBySeqAligned_Arr = StateSeqUtil.convertStateSeq_list2MAT(
            zHatBySeqAligned)

        MATVarsDict = dict(zHatBySeqAligned=zHatBySeqAligned_Arr)
        matfilepath = os.path.join(savedir,
                                   prefix + 'MAPStateSeqsAligned.mat')
        scipy.io.savemat(matfilepath, MATVarsDict, oned_as='row')

        kwargs['Data'] = Data
        calcHammingDistanceAndSave(zHatFlatAligned, **kwargs)


def calcHammingDistanceAndSave(zHatFlatAligned,
                               excludeTstepsWithNegativeTrueLabels=1,
                               **kwargs):
    ''' Calculate hamming distance for all sequences, saving to flat file.

    Excludes any

    Keyword Args (all workspace variables passed along from learning alg)
    -------
    hmodel : current HModel object
    Data : current Data object
        representing *entire* dataset (not just one chunk)

    Returns
    -------
    None. Hamming distance saved to file.

    Output
    -------
    hamming-distance.txt
    '''
    Data = kwargs['Data']
    zTrue = Data.TrueParams['Z']
    hdistance = StateSeqUtil.calcHammingDistance(
        zTrue,
        zHatFlatAligned,
        **kwargs)
    normhdist = float(hdistance) / float(zHatFlatAligned.size)

    learnAlgObj = kwargs['learnAlg']
    lapFrac = kwargs['lapFrac']
    prefix = makePrefixForLap(lapFrac)
    outpath = os.path.join(learnAlgObj.savedir, 'hamming-distance.txt')
    with open(outpath, 'a') as f:
        f.write('%.6f\n' % (normhdist))
