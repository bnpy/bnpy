from builtins import *
import numpy as np
import argparse

import bnpy.mergemove.MLogger as MLogger
from bnpy.ioutil.DataReader import loadDataFromSavedTask, loadLPKwargsFromDisk
from bnpy.ioutil.DataReader import loadKwargsFromDisk
from bnpy.ioutil.ModelReader import loadModelForLap

def tryMergeProposalForSpecificTarget(
        Data, hmodel,
        LPkwargs=dict(),
        kA=0,
        kB=1,
        verbose=True,
        doHeuristicUpdateRho=False,
        **kwargs):
    ''' Execute merge for specific whole dataset

    Returns
    -------
    propModel : HModel
    propSS : SuffStatBag
    propLscore : scalar real
        ELBO score of proposed model
    curModel : HModel
    curSS : SuffStatBag
    curLscore : scalar real
        ELBO score of current model
    '''
    curModel = hmodel.copy()
    propModel = hmodel.copy()

    # Update current
    curLP = curModel.calc_local_params(Data, **LPkwargs)
    curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=1)
    curModel.update_global_params(curSS)
    curLdict = curModel.calc_evidence(SS=curSS, todict=1)
    curLscore = curLdict['Ltotal']

    scaleF = curModel.obsModel.getDatasetScale(curSS)

    def makePairsForComp(k):
        pairList = [(kLower, k) for kLower in range(0, k)]
        pairList.extend([(k, kUpper) for kUpper in range(k+1, curSS.K)])
        return pairList

    def makeListOfBestPairsForCompByObsGap(k):
        pairList = makePairsForComp(k)
        gapVec = curModel.obsModel.calcHardMergeGap_SpecificPairs(curSS,
            makePairsForComp(k))
        gapVec /= scaleF
        bestLocs = np.argsort(-1 * gapVec)
        print("Best pairs for cluster %d" % (k))
        for ii in bestLocs[:5]:
            a, b = pairList[ii]
            if k == a:
                kpartner = b
            else:
                kpartner = a
            print("   kpartner %3d:  % .3e" % (kpartner, gapVec[ii]))

    makeListOfBestPairsForCompByObsGap(kA)
    makeListOfBestPairsForCompByObsGap(kB)

    oGap = curModel.obsModel.calcHardMergeGap(curSS, kA=kA, kB=kB) / scaleF
    aGap, propRho, propOmega = curModel.allocModel.calcHardMergeGap(curSS, kA=kA, kB=kB, returnRhoOmega=1)
    aGap /= scaleF

    print("% .3e   obsModel estimated-gain" % (oGap))
    print("% .3e allocModel estimated-gain" % (aGap))
    print("% .3e      total estimated-gain" % (aGap + oGap))

    # Update proposal
    if curModel.getAllocModelName().count('DPMixture'):
        propResp = np.delete(curLP['resp'], kB, axis=1)
        propResp[:, kA] += curLP['resp'][:, kB]
        assert np.allclose(1.0, propResp.sum(axis=1))
        propLP = dict(resp=propResp)
    elif curModel.getAllocModelName().count('HDPTopic'):
        propResp = np.delete(curLP['resp'], kB, axis=1)
        propResp[:, kA] += curLP['resp'][:, kB]
        propLP = curModel.allocModel.initLPFromResp(
            Data, dict(resp=propResp))
    else:
        raise ValueError("Unrecognized getAllocModelName")

    propSS = propModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)

    propModel.obsModel.update_global_params(propSS)
    if doHeuristicUpdateRho:
        propModel.allocModel.rho = propRho
        propModel.allocModel.omega = propOmega
        propModel.allocModel.K = propSS.K
    else:
        propModel.allocModel.update_global_params(propSS)
    print("propOmega:")
    print(' '.join(['%.1f' % (x) for x in propModel.allocModel.omega[:5]]))

    propLdict = propModel.calc_evidence(SS=propSS, todict=1)
    propLscore = propLdict['Ltotal']

    if verbose:
        print("Merging cluster %d and %d ..." % (kA, kB))
        if propLscore - curLscore > 0:
            print("  ACCEPTED")
        else:
            print("  REJECTED")
        print("% .4e  cur ELBO score" % (curLscore))
        print("% .4e prop ELBO score" % (propLscore))
        print("Change in ELBO score: %.4e" % (propLscore - curLscore))
        print("")
        for key in sorted(curLdict.keys()):
            if key.count('_') or key.count('total'):
                continue
            print("  gain %8s % .3e" % (
                key, propLdict[key] - curLdict[key]))

    return (
        propModel,
        propSS,
        propLscore,
        curModel,
        curSS,
        curLscore)


def tryMergeProposalForSavedTask(
        taskoutpath=None,
        lap=None,
        lapFrac=0,
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

    hmodel, lapFrac = loadModelForLap(taskoutpath, lapFrac)
    Data = loadDataFromSavedTask(taskoutpath, batchID=batchID)
    kwargs['LPkwargs'] = loadLPKwargsFromDisk(taskoutpath)

    tryMergeProposalForSpecificTarget(
        Data, hmodel,
        **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskoutpath', type=str)
    parser.add_argument('--lap', type=float, default=None)
    parser.add_argument('--lapFrac', type=float, default=None)
    parser.add_argument('--outputdir', type=str, default='/tmp/')
    parser.add_argument('--kA', type=int, default=0)
    parser.add_argument('--kB', type=int, default=1)
    parser.add_argument('--batchID', type=int, default=None)
    parser.add_argument('--verbose', type=int, default=True)
    parser.add_argument('--doHeuristicUpdateRho', type=int, default=0)
    args = parser.parse_args()

    MLogger.configure(args.outputdir,
        doSaveToDisk=0,
        doWriteStdOut=1)
    tryMergeProposalForSavedTask(**args.__dict__)
