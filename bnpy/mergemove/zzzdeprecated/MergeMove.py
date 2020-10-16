import numpy as np

import MergeLogger
ELBO_GAP_ACCEPT_TOL = 1e-6


def run_many_merge_moves(curModel, curSS, curELBO, mPairIDs, M=None,
                         logFunc=MergeLogger.log,
                         isBirthCleanup=0,
                         **kwargs):
    ''' Run many pre-selected merge moves, keeping all that improve ELBO.

    Returns
    --------
    model : new bnpy HModel
    SS : bnpy SuffStatBag
    ELBO : float
    MergeInfo : dict with info about all accepted merges
    '''
    # eligibleIDs : list from 0, 1, ... len(mPairIDs)
    #  provides index of which original candidate we are now attempting
    eligibleIDs = list(range(len(mPairIDs)))

    CompIDShift = np.zeros(curSS.K, dtype=np.int32)

    if 'mergePerLap' in kwargs:
        nMergeTrials = kwargs['mergePerLap']
    else:
        nMergeTrials = len(mPairIDs)

    trialID = 0
    AcceptedPairs = list()
    AcceptedPairOrigIDs = list()
    ELBOGain = 0
    nSkip = 0

    sF = curModel.obsModel.getDatasetScale(curSS)
    isHDPTopicModel = str(type(curModel.allocModel)).count('HDPTopic') > 0
    if len(mPairIDs) > 0 and isHDPTopicModel:
        aList, bList = list(zip(*mPairIDs))
        OGapMat = np.zeros((curSS.K, curSS.K))
        OGapList = curModel.obsModel.calcHardMergeGap_SpecificPairs(
            curSS, mPairIDs)
        OGapMat[aList, bList] = OGapList / sF

    while trialID < nMergeTrials and len(eligibleIDs) > 0:
        if len(eligibleIDs) == 0:
            break
        curID = eligibleIDs.pop(0)

        # kA, kB are the "original" indices, under input model with K comps
        kA, kB = mPairIDs[curID]
        assert kA < kB

        if CompIDShift[kA] == -1 or CompIDShift[kB] == -1:
            nSkip += 1
            continue

        # jA, jB are "shifted" indices under our new model, with K- Kaccepted
        # comps
        jA = kA - CompIDShift[kA]
        jB = kB - CompIDShift[kB]

        if M is not None:
            Mcand = M[jA, jB]
            scoreMsg = '%9.5f' % (M[jA, jB])
        else:
            Mcand = None
            scoreMsg = ''
        Nvec = curSS.getCountVec()
        scoreMsg += " %5d %5d" % (Nvec[jA], Nvec[jB])

        """
        # Extra diagnostics for HDPTopic models
        if isHDPTopicModel:
            scoreMsg += " % .7e" % (OGapMat[kA, kB])

            EntropyGap = curSS.getELBOTerm('ElogqZ')[[jA, jB]].sum() \
                - curSS.getMergeTerm('ElogqZ')[jA, jB]
            scoreMsg += " % .7e" % (EntropyGap / sF)

            cDirThetaGap = curSS.getMergeTerm('gammalnTheta')[jA, jB] \
                - curSS.getELBOTerm('gammalnTheta')[[jA, jB]].sum()
            scoreMsg += " % .7e" % (cDirThetaGap / sF)
        """
        curModel, curSS, curELBO, MoveInfo = buildCandidateAndKeepIfImproved(
            curModel, curSS, curELBO, jA, jB, Mcand, **kwargs)
        logFunc('%3d | %3d %3d | %d % .7e %s'
                % (trialID, kA, kB,
                    MoveInfo['didAccept'], MoveInfo['ELBOGain'], scoreMsg),
                'debug')
        if MoveInfo['didAccept']:
            CompIDShift[kA] = -1
            CompIDShift[kB] = -1
            offIDs = CompIDShift < 0
            CompIDShift[kB + 1:] += 1
            CompIDShift[offIDs] = -1

            AcceptedPairs.append((jA, jB))
            AcceptedPairOrigIDs.append((kA, kB))
            ELBOGain += MoveInfo['ELBOGain']

            # Update PairScoreMatrix
            if M is not None:
                M = np.delete(np.delete(M, jB, axis=0), jB, axis=1)
                M[jA, jA + 1:] = 0
                M[:jA, jA] = 0
        else:
            if M is not None:
                M[jA, jB] = MoveInfo['ELBOGain']
        trialID += 1

    if len(AcceptedPairs) > 0:
        msg = 'ev increased % .4e' % (ELBOGain)
    else:
        msg = ''

    msg = 'MERGE %d/%d accepted. %s' % (len(AcceptedPairs), trialID, msg)
    if isBirthCleanup:
        logFunc(msg, 'debug')
    else:
        logFunc(msg)

    if not isBirthCleanup:
        logFunc('MERGE %d pairs skipped to due previous accepted merge.'
                % (nSkip), 'debug')
    Info = dict(AcceptedPairs=AcceptedPairs,
                AcceptedPairOrigIDs=AcceptedPairOrigIDs,
                ELBOGain=ELBOGain,
                ScoreMat=M)
    return curModel, curSS, curELBO, Info


def buildCandidateAndKeepIfImproved(curModel, curSS, curELBO, kA, kB,
                                    Mcur=0,
                                    **kwargs):
    ''' Create candidate with kA, kB merged, and keep if ELBO improves.
    '''
    if 'mergeUpdateFast' not in kwargs:
        kwargs['mergeUpdateFast'] = 1
    assert np.isfinite(curELBO)

    # Rewrite candidate's kA component to be the merger of kA+kB
    propSS = curSS.copy()
    propSS.mergeComps(kA, kB)
    assert propSS.K == curSS.K - 1

    propModel = curModel.copy()
    if kwargs['mergeUpdateFast']:
        propModel.update_global_params(propSS, mergeCompA=kA, mergeCompB=kB)
    else:
        propModel.update_global_params(propSS)

    # After update_global_params, propModel's comps exactly match propSS's.
    # So at this point, kB has been deleted, and propModel has K-1 components.
    assert propModel.obsModel.K == curModel.obsModel.K - 1
    assert propModel.allocModel.K == curModel.allocModel.K - 1

    # Verify Merge improves the ELBO
    propELBO = propModel.calc_evidence(SS=propSS)
    assert np.isfinite(propELBO)

    didAccept = propELBO > curELBO - ELBO_GAP_ACCEPT_TOL
    Info = dict(didAccept=didAccept,
                ELBOGain=propELBO - curELBO,
                )

    if didAccept:
        return propModel, propSS, propELBO, Info
    else:
        return curModel, curSS, curELBO, Info
