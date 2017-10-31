"""
Functions that evaluate delete proposals and decide to accept/reject.

- runDeleteMove
"""
import numpy as np
import DeleteLogger
from DCollector import hasValidKey, getSize


def runDeleteMoveAndUpdateMemory(curModel, curSS, Plan,
                                 nRefineIters=2,
                                 LPkwargs=None,
                                 SSmemory=None,
                                 Kmax=np.inf,
                                 lapFrac=None,
                                 deleteNontargetStrategy='forget',
                                 doVizDelete=0,
                                 **kwargs):
    """ Propose model with fewer comps and accept if ELBO improves.

    Will update the memoized suff stats for each batch (SSmemory)
    in place to reflect any accepted deletions.

    Returns
    --------
    bestModel : HModel, with K' states
    bestSS : SuffStatBag with K' states
    SSmemory : dict of suff stats, one per batch with K' states
    Plan : dict, with updated fields
    * didAccept
    * acceptedUIDs
    * acceptedIDs

    Post Condition
    --------
    SSmemory has valid stats for each batch under proposed model
    with K' states. Summing over all entries of SSmemory
    will be exactly equal to the whole-dataset stats bestSS.

    bestSS and each entry of SSmemory have NO ELBO or Merge terms.
    """

    if lapFrac is not None:
        msg = '<<<<<<<<<<<<<<<<<<<< DEvaluator.runDeleteMove @ lap %6.2f' \
            % (np.ceil(lapFrac))
        DeleteLogger.log(msg)
        DeleteLogger.log('Target Size: %d' % (getSize(Plan['DTargetData'])))

    if SSmemory is None:
        SSmemory = dict()
    if LPkwargs is None:
        LPkwargs = dict()
    if curSS.K == 1:
        Plan['didAccept'] = 0
        Plan['acceptedUIDs'] = list()
        if lapFrac is not None:
            DeleteLogger.log('ABANDONED. Cannot delete when K=1.')
        return curModel, curSS, SSmemory, Plan

    # bestModel, bestSS represent best so far
    bestModel = curModel
    bestSS = curSS
    besttargetSS = Plan['targetSS']
    assert np.allclose(besttargetSS.uIDs, bestSS.uIDs)

    # Calculate the current ELBO score
    targetData = Plan['DTargetData']
    totalScale = curModel.obsModel.getDatasetScale(curSS)
    bestELBOobs = curModel.obsModel.calcELBO_Memoized(curSS)
    bestELBOalloc = curModel.allocModel.calcELBO_LinearTerms(SS=curSS)

    bestELBOobs /= totalScale
    bestELBOalloc /= totalScale

    totalELBOImprovement = 0
    didAccept = 0
    acceptedUIDs = list()
    acceptedPairs = list()
    for delCompUID in Plan['candidateUIDs']:

        if lapFrac is not None:
            DeleteLogger.log('Deleting UID %d... ' % (delCompUID))

        if bestSS.K == 1:
            if lapFrac is not None:
                DeleteLogger.log('  skipped. Cannot delete when K=1.')
            continue  # Don't try to remove the final comp!

        delCompID = np.flatnonzero(bestSS.uIDs == delCompUID)[0]

        # Construct candidate model and suff stats
        propModel = bestModel.copy()
        ptargetSS = besttargetSS.copy()
        propSS = bestSS.copy()
        propSS.setMergeFieldsToZero()

        if deleteNontargetStrategy == 'forget':
            propSS.removeComp(delCompID)
            ptargetSS.removeComp(delCompID)
            propModel.update_global_params(propSS)
            ELBOGain_NonLinear_nontarget = 0

        elif deleteNontargetStrategy == 'merge':
            raise NotImplementedError('TODO')
            '''
            # Make sure we haven't used this comp already
            # in a previous delete
            usedBefore = False
            for kA, kB in acceptedPairs:
                if delCompID == kA:
                    usedBefore = True
                elif delCompID == kB:
                    usedBefore = True
            if usedBefore:
                continue  # move on to next delCompID in Plan

            # Remove all target stats,
            # so propSS represents only remaining data items
            propSS -= ptargetSS
            propModel.update_global_params(propSS)

            mPairIDs = makeMPairIDsWith(delCompID, propSS.K)
            kA, kB = propModel.getBestMergePair(propSS, mPairIDs)
            # Compute ELBO gap for cached terms under proposed merge
            ELBOgap_cached_rest = propModel.allocModel.\
                calcCachedELBOGap_SinglePair(
                    propSS, kA, kB, delCompID=delCompID) / totalScale

            ELBOTerms = propModel.allocModel.\
                calcCachedELBOTerms_SinglePair(
                    propSS, kA, kB, delCompID=delCompID)
            propSS.mergeComps(kA, kB)
            for key, arr in ELBOTerms.items():
                propSS.setELBOTerm(key, arr, propSS._ELBOTerms._FieldDims[key])

            # Remove delCompID from target stats
            ptargetSS.removeComp(delCompID)
            # Add target stats back into propSS, so represents whole dataset
            propSS += ptargetSS
            propModel.update_global_params(propSS)

            # Verify construction via the merge of all non-target entities
            propCountPlusTarget = propSS.getCountVec().sum() \
                + besttargetSS.getCountVec()[delCompID]
            bestCount = bestSS.getCountVec().sum()
            assert np.allclose(propCountPlusTarget, bestCount)

            # Pretend we are just doing a hard merge...
            mergeSS = bestSS.copy()
            mergeModel = bestModel.copy()
            mELBOTerms = mergeModel.allocModel.\
                calcCachedELBOTerms_SinglePair(
                    mergeSS, kA, kB, delCompID=delCompID)
            mergeSS.mergeComps(kA, kB)
            for key, arr in mELBOTerms.items():
                mergeSS.setELBOTerm(
                    key,
                    arr,
                    propSS._ELBOTerms._FieldDims[key])
            mergeModel.update_global_params(mergeSS)
            mergeGap = mergeModel.calc_evidence(SS=mergeSS) - \
                bestModel.calc_evidence(SS=bestSS)

            if 'WholeDataset' in Plan:
                Data = Plan['WholeDataset']
                remUnitIDs = np.setdiff1d(np.arange(Data.get_size()),
                                          Plan['dataUnitIDs'])
                remData = Data.select_subset_by_mask(remUnitIDs)

                remLP = bestModel.calc_local_params(remData)
                remSS = bestModel.get_global_suff_stats(
                    remData, remLP, doPrecompEntropy=1,
                    doPrecompMergeEntropy=1, mPairIDs=[(kA, kB)])
                Hvec = -1 * remSS.getELBOTerm('ElogqZ')
                mHvec = np.delete(Hvec, kB)
                mHvec[kA] = -1 * remSS.getMergeTerm('ElogqZ')[kA, kB]
                tight_ELBOgap_rest = (mHvec.sum() - Hvec.sum()) / totalScale

                print "Hrem fast   %.4f" % (ELBOgap_cached_rest)
                print "Hrem tight  %.4f" % (tight_ELBOgap_rest)
            '''
        else:
            msg = 'Unrecognised deleteNontargetStrategy: %s' \
                % (deleteNontargetStrategy)
            raise ValueError(msg)

        # Refine candidate with local/global steps
        didAcceptCur = 0
        for riter in range(nRefineIters):
            ptargetLP = propModel.calc_local_params(targetData, **LPkwargs)
            propSS -= ptargetSS
            ptargetSS = propModel.get_global_suff_stats(targetData, ptargetLP,
                                                        doPrecompEntropy=1)
            propSS += ptargetSS
            propModel.update_global_params(propSS)

            propELBOobs = propModel.\
                obsModel.calcELBO_Memoized(propSS) / totalScale
            propELBOalloc = propModel.\
                allocModel.calcELBO_LinearTerms(SS=propSS) / totalScale

            propNLELBO = \
                propModel.allocModel.calcELBO_NonlinearTerms(SS=ptargetSS)
            curNLELBO = \
                bestModel.allocModel.calcELBO_NonlinearTerms(SS=besttargetSS)
            ELBOGain_NonLinear_target = (propNLELBO - curNLELBO) / totalScale

            ELBOGain = propELBOobs - bestELBOobs \
                + propELBOalloc - bestELBOalloc \
                + ELBOGain_NonLinear_nontarget \
                + ELBOGain_NonLinear_target

            if not np.isfinite(ELBOGain):
                break
            if ELBOGain > 0 or bestSS.K > Kmax:
                didAcceptCur = 1
                didAccept = 1
                break

        # Log result of this proposal
        curMsg = makeLogMessage(bestSS, besttargetSS,
                                label='cur', compUID=delCompUID)
        propMsg = makeLogMessage(propSS, ptargetSS,
                                 label='prop', compUID=delCompUID)
        resultMsg = makeLogMessageForResult(
            ELBOGain, didAcceptCur,
            ELBOgap_alloc=propELBOalloc - bestELBOalloc,
            ELBOgap_obs=propELBOobs - bestELBOobs,
            ELBOgap_Htrgt=ELBOGain_NonLinear_target,
            ELBOgap_Hrest=ELBOGain_NonLinear_nontarget)
        if doVizDelete:
            levelStr = 'info'
        else:
            levelStr = 'debug'
        if lapFrac is not None:
            DeleteLogger.log(curMsg, levelStr)
            DeleteLogger.log(propMsg, levelStr)
            DeleteLogger.log(resultMsg, levelStr)

        if doVizDelete:
            from bnpy.viz.PlotComps import plotCompsFromHModel
            from matplotlib import pylab

            if deleteNontargetStrategy == 'merge':
                if delCompID == kA:
                    otherID = kB
                else:
                    otherID = kA
                compsToHighlight = [delCompID, otherID]
            else:
                compsToHighlight = [delCompID]
            plotCompsFromHModel(bestModel, Data=targetData,
                                compsToHighlight=compsToHighlight)
            pylab.show(block=0)
            input('Press any key to continue>>>')
            pylab.close()

        # Update best model/stats to accepted values
        if didAcceptCur:
            totalELBOImprovement += ELBOGain
            acceptedUIDs.append(delCompUID)
            bestELBOobs = propELBOobs
            bestELBOalloc = propELBOalloc
            bestModel = propModel
            besttargetLP = ptargetLP
            besttargetSS = ptargetSS
            bestSS = propSS
            bestSS.setMergeFieldsToZero()
            if deleteNontargetStrategy == 'merge':
                acceptedPairs.append((kA, kB))

        # << end for loop over each candidate comp

    Plan['didAccept'] = didAccept
    Plan['acceptedUIDs'] = acceptedUIDs

    Plan['nAccept'] = len(acceptedUIDs)
    Plan['nTotal'] = len(Plan['candidateUIDs'])
    # Update SSmemory to reflect accepted deletes
    if didAccept:
        bestSS.setELBOFieldsToZero()
        bestSS.setMergeFieldsToZero()
        for batchID in SSmemory:
            SSmemory[batchID].setELBOFieldsToZero()
            SSmemory[batchID].setMergeFieldsToZero()

            if hasValidKey(Plan, 'targetSSByBatch'):
                doEditBatch = batchID in Plan['targetSSByBatch']

            # Decrement : subtract old value of targets in this batch
            # Here, SSmemory has K states
            if doEditBatch:
                SSmemory[batchID] -= Plan['targetSSByBatch'][batchID]

            # Update batch-specific stats with accepted deletes
            if deleteNontargetStrategy == 'merge':
                for (kA, kB) in acceptedPairs:
                    SSmemory[batchID].mergeComps(kA, kB)
            else:
                for uID in acceptedUIDs:
                    kk = np.flatnonzero(SSmemory[batchID].uIDs == uID)[0]
                    SSmemory[batchID].removeComp(kk)

            assert np.allclose(SSmemory[batchID].uIDs, bestSS.uIDs)
            assert SSmemory[batchID].K == besttargetLP['resp'].shape[1]
            assert SSmemory[batchID].K == bestModel.allocModel.K
            assert SSmemory[batchID].K == bestSS.K

            # Increment : add in new value of targets in this batch
            # Here, SSmemory has K-1 states
            if doEditBatch:
                relUnitIDs = np.flatnonzero(Plan['batchIDs'] == batchID)
                Data_b = targetData.select_subset_by_mask(
                    relUnitIDs, doTrackFullSize=False)
                targetLP_b = bestModel.allocModel.selectSubsetLP(
                    targetData,
                    besttargetLP,
                    relUnitIDs)
                targetSS_b = bestModel.get_global_suff_stats(
                    Data_b, targetLP_b)
                SSmemory[batchID] += targetSS_b

            SSmemory[batchID].setELBOFieldsToZero()
            SSmemory[batchID].setMergeFieldsToZero()

    return bestModel, bestSS, SSmemory, Plan


def makeMPairIDsWith(k, K, excludeIDs=None):
    """ Create list of possible merge pairs including k, excluding excludeIDs.

    Returns
    -------
    mPairIDs : list of tuples
        each entry is a pair (kA, kB) satisfying kA < kB < K
    """
    if excludeIDs is None:
        excludeIDs = list()
    mPairIDs = list()
    for j in range(K):
        if j in excludeIDs:
            continue
        elif j == k:
            continue
        elif j < k:
            mPairIDs.append((j, k))
        else:
            mPairIDs.append((k, j))
    return mPairIDs


def makeLogMessageForResult(ELBOGain, didAccept=0,
                            ELBOgap_alloc=0,
                            ELBOgap_Htrgt=0,
                            ELBOgap_Hrest=0,
                            ELBOgap_obs=0):
    if didAccept:
        label = ' ACCEPTED '
    else:
        label = ' rejected '
    label += ' ELBOGain %10.8f' % (ELBOGain)
    label += '  alloc %10.8f' % (ELBOgap_alloc)
    label += '  obs %10.8f' % (ELBOgap_obs)
    label += '  Htrgt %10.8f' % (ELBOgap_Htrgt)
    label += '  Hrest %10.8f' % (ELBOgap_Hrest)
    return label


def makeLogMessage(aggSS, targetSS,
                   label='cur',
                   compUID=0):

    if label.count('cur'):
        label = " compUID %3d  " % (compUID) + label
    else:
        label = '             ' + label

    msg = '%s K=%3d | aggN %10.1f | trgtN %10.1f' \
          % (label,
             targetSS.K,
             aggSS.getCountVec().sum(),
             targetSS.getCountVec().sum())

    if label.count('cur'):
        k = np.flatnonzero(aggSS.uIDs == compUID)[0]
        msg += " | aggN[k] %10.1f | trgtN[k] %10.1f" \
               % (aggSS.getCountVec()[k], targetSS.getCountVec()[k])
    else:
        msg = msg.replace('aggN', '    ')
        msg = msg.replace('trgtN', '     ')

    return msg
