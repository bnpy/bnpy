from builtins import *
import argparse
import numpy as np
import os

from bnpy.ioutil.DataReader import loadDataFromSavedTask, loadLPKwargsFromDisk
from bnpy.ioutil.DataReader import loadKwargsFromDisk
from bnpy.ioutil.ModelReader import loadModelForLap
from bnpy.util import StateSeqUtil
from bnpy.birthmove.BCreateOneProposal import \
    makeSummaryForBirthProposal_HTMLWrapper
import bnpy.birthmove.BLogger as BLogger

DefaultBirthArgs = dict(
    Kmax=100,
    b_nStuckBeforeQuit=10,
    b_creationProposalName='bregmankmeans',
    b_Kfresh=10,
    b_nRefineSteps=10,
    b_NiterForBregmanKMeans=1,
    b_minRespForEachTargetAtom=0.1,
    b_minNumAtomsInEachTargetDoc=50,
    b_minNumAtomsForNewComp=1,
    b_minNumAtomsForTargetComp=2,
    b_minPercChangeInNumAtomsToReactivate=0.01,
    b_cleanupWithMerge=0,
    b_cleanupMaxNumMergeIters=10,
    b_cleanupMaxNumAcceptPerIter=1,
    b_debugOutputDir='/tmp/',
    b_debugWriteHTML=1,
    b_method_xPi='normalized_counts',
    b_method_initCoordAscent='fromprevious',
    b_method_doInitCompleteLP=1,
    b_localStepSingleDoc='fast',
    )

def tryBirthForTask(
        taskoutpath=None,
        lap=None, lapFrac=0,
        targetUID=0,
        batchID=None,
        **kwargs):
    '''

    Post Condition
    --------------
    * Logging messages are printed.
    * HTML report is saved.
    '''
    if lap is not None:
        lapFrac = lap

    curModel, lapFrac = loadModelForLap(taskoutpath, lapFrac)
    Data = loadDataFromSavedTask(taskoutpath, batchID=batchID)

    LPkwargs = loadLPKwargsFromDisk(taskoutpath)
    SavedBirthKwargs = loadKwargsFromDisk(taskoutpath, 'args-birth.txt')

    if targetUID < 0:
        targetUID = findCompInModelWithLargestMisalignment(curModel, Data)

    BirthArgs = dict(**DefaultBirthArgs)
    BirthArgs.update(SavedBirthKwargs)
    for key, val in list(kwargs.items()):
        if val is not None:
            BirthArgs[key] = val
            print('%s: %s' % (key, str(val)))

    curLP = curModel.calc_local_params(Data, **LPkwargs)
    curSS = curModel.get_global_suff_stats(
        Data, curLP,
        trackDocUsage=1, doPrecompEntropy=1, trackTruncationGrowth=1)
    curLscore = curModel.calc_evidence(SS=curSS)

    print("Target UID: %d" % (targetUID))
    print("Current count: %.2f" % (curSS.getCountForUID(targetUID)))

    xSS = makeSummaryForBirthProposal_HTMLWrapper(
        Data, curModel, curLP,
        curSSwhole=curSS,
        targetUID=int(targetUID),
        newUIDs=list(range(curSS.K, curSS.K + int(BirthArgs['b_Kfresh']))),
        LPkwargs=LPkwargs,
        lapFrac=lapFrac,
        dataName=Data.name,
        **BirthArgs)

    '''
    propModel, propSS = createBirthProposal(curModel, SS, xSS)
    didAccept, AcceptInfo = evaluateBirthProposal(
        curModel=curModel, curSS=curSS, propModel=propModel, propSS=propSS)
    '''

def findCompInModelWithLargestMisalignment(model, Data, Zref=None):
    ''' Finds cluster in model that is best candidate for a birth move.

    Post Condition
    --------------
    Prints useful info to stdout.
    '''
    if Zref is None:
        Zref = Data.TrueParams['Z']
    LP = model.calc_local_params(Data)
    Z = LP['resp'].argmax(axis=1)
    AZ, AlignInfo = StateSeqUtil.alignEstimatedStateSeqToTruth(
        Z, Zref, returnInfo=1)
    maxK = AZ.max()
    dist = np.zeros(maxK)
    for k in range(maxK):
        mask = AZ == k
        nDisagree = np.sum(Zref[mask] != k)
        nTotal = mask.sum()
        dist[k] = float(nDisagree) / (float(nTotal) + 1e-10)
        print(k, dist[k])
    ktarget = np.argmax(dist)
    korig = AlignInfo['AlignedToOrigMap'][ktarget]
    print('ktarget %d: %s' % (ktarget, chr(65+ktarget)))
    print('korig %d' % (korig))
    # Determine what is hiding inside of it that shouldnt be
    mask = AZ == ktarget
    nTarget = np.sum(mask)
    print('%d total atoms assigned to ktarget...' % (nTarget))
    trueLabels = np.asarray(np.unique(Zref[mask]), np.int32)
    for ll in trueLabels:
        nTrue = np.sum(Zref[mask] == ll)
        print('%d/%d should have true label %d: %s' % (
            nTrue, nTarget, ll, chr(65+ll)))
    return korig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskoutpath', type=str)
    parser.add_argument('--lap', type=float, default=None)
    parser.add_argument('--lapFrac', type=float, default=None)
    parser.add_argument('--outputdir', type=str, default='/tmp/')
    parser.add_argument('--targetUID', type=int, default=0)
    parser.add_argument('--batchID', type=int, default=None)
    for key, val in list(DefaultBirthArgs.items()):
        parser.add_argument('--' + key, type=type(val), default=None)
    args = parser.parse_args()

    BLogger.configure(args.outputdir,
        doSaveToDisk=0,
        doWriteStdOut=1,
        stdoutLevel=0)
    tryBirthForTask(**args.__dict__)
