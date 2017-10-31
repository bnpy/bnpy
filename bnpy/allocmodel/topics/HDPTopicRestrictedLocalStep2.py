from builtins import *
import numpy as np

from scipy.special import digamma, gammaln
from bnpy.util import NumericUtil
from bnpy.allocmodel import make_xPiVec_and_emptyPi


def calcDocTopicCountCorrelationFromTargetToAbsorbingSet(
        DocTopicMat, ktarget, kabsorbList, MINVAL=1.0e-8):
    ''' Find correlation in DocTopicCount between target and absorbing states.

    Returns
    -------
    CorrVec : 1D array, size nAbsorbing
        CorrVec[j] : correlation value (-1 < corr < 1)
            from kabsorbList[j] to the target
    '''
    D = DocTopicMat.shape[0]
    Smat = np.dot(DocTopicMat.T, DocTopicMat)
    svec = np.sum(DocTopicMat, axis=0)

    nanIDs = np.isnan(Smat)
    Smat[nanIDs] = 0
    svec[np.isnan(svec)] = 0
    offlimitcompIDs = np.logical_or(np.isnan(svec), svec < MINVAL)
    CovMat = Smat / D - np.outer(svec / D, svec / D)
    varc = np.diag(CovMat)
    sqrtc = np.sqrt(varc)
    sqrtc[offlimitcompIDs] = MINVAL
    assert sqrtc.min() >= MINVAL
    CorrMat = CovMat / np.outer(sqrtc, sqrtc)
    return CorrMat[kabsorbList, ktarget].copy()


def summarizeRestrictedLocalStep_HDPTopicModel(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        curSSwhole=None,
        targetUID=None,
        ktarget=None,
        kabsorbList=None,
        xUIDs=None,
        xObsModel=None,
        xInitSS=None,
        doBuildOnInit=False,
        xPiVec=None,
        curPiVec=None,
        emptyPiFrac=0.0,
        b_emptyPiFrac=None,
        nUpdateSteps=5,
        d_initWordCounts='none',
        **kwargs):
    ''' Perform restricted local step and summarize it.

    Returns
    -------
    xSSslice : SuffStatBag
    Info : dict with other information
    '''
    if b_emptyPiFrac is not None:
        emptyPiFrac = b_emptyPiFrac

    # Translate specififed unique-IDs (UID) into current order IDs
    if targetUID is not None:
        ktarget = curSSwhole.uid2k(targetUID)

    Kfresh = None
    if xUIDs is not None:
        Kfresh = len(xUIDs)
        intersectUIDs = np.intersect1d(xUIDs, curSSwhole.uids)
        if intersectUIDs.size > 0:
            kabsorbList = list()
            for uid in intersectUIDs:
                kabsorbList.append(curSSwhole.uid2k(uid))
            kabsorbList.sort()

    if kabsorbList is not None:
        if Kfresh is None:
            Kfresh = len(kabsorbList)
        else:
            assert len(kabsorbList) == Kfresh
    if xInitSS is not None:
        if Kfresh is None:
            Kfresh = xInitSS.K
        else:
            assert Kfresh == xInitSS.K
    if xObsModel is not None:
        if Kfresh is None:
            Kfresh = xObsModel.K
        else:
            assert Kfresh == xObsModel.K

    # Identify original cluster probabilities for all clusters involved
    # And maintain sum constraints within these for proposed state
    if curPiVec is None:
        curPiVec = curModel.allocModel.get_active_comp_probs()
    if kabsorbList is None:
        availablePiMass = curPiVec[ktarget]
        emptyPi = emptyPiFrac * availablePiMass
    else:
        availablePiMass = np.sum(curPiVec[kabsorbList]) + curPiVec[ktarget]
        emptyPi = 0.0

    # Create probabilities for each of the Kfresh new clusters
    # by subdividing the involved clusters' original probabilities
    if xPiVec is None:
        if kabsorbList is None:
            xPiVec = ((1-emptyPiFrac) * availablePiMass) / Kfresh * np.ones(Kfresh)
        else:
            xPiVec = curPiVec[kabsorbList].copy()
            xPiVec /= xPiVec.sum()
            xPiVec *= availablePiMass

    assert np.allclose(availablePiMass, emptyPi + np.sum(xPiVec))
    assert Kfresh == xPiVec.size

    # Create expansion observation model, if necessary
    if xObsModel is None:
        # # Create expanded observation model
        xObsModel = curModel.obsModel.copy()
        # # Verify provided initialization-stats exist
        # # Otherwise, there's no way to initialize the new obsmodel
        assert xInitSS is not None
        # # Optionally, for delete moves smartly initialize absorbing clusters
        if kabsorbList is not None:
            isMult = curModel.getObsModelName().count('Mult')
            if not doBuildOnInit and isMult and d_initWordCounts.count('corr'):
                corrVec = calcDocTopicCountCorrelationFromTargetToAbsorbingSet(
                    curLPslice['DocTopicCount'], ktarget, kabsorbList)
                bestAbsorbIDs = np.flatnonzero(corrVec >= .001)
                for k in bestAbsorbIDs:
                    xInitSS.WordCounts[k,:] += curSSwhole.WordCounts[ktarget,:]
        # # Initialize xObsModel using provided stats
        xObsModel.update_global_params(xInitSS)
    assert Kfresh == xObsModel.K

    # Perform restricted inference!
    # xLPslice contains local params for all Kfresh expansion clusters
    xalphaPi = curModel.allocModel.alpha * xPiVec
    xLPslice = restrictedLocalStep_HDPTopicModel(
        Dslice=Dslice,
        curLPslice=curLPslice,
        ktarget=ktarget,
        kabsorbList=kabsorbList,
        xObsModel=xObsModel,
        xalphaPi=xalphaPi,
        thetaEmptyComp=curModel.allocModel.alpha * emptyPi,
        nUpdateSteps=nUpdateSteps,
        doBuildOnInit=doBuildOnInit,
        xInitSS=xInitSS,
        **kwargs)
    if emptyPiFrac > 0:
        assert "HrespOrigComp" in xLPslice

    # Summarize this expanded local parameter pack
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        trackDocUsage=1, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    if xUIDs is not None:
        xSSslice.setUIDs(xUIDs)
    assert xSSslice.hasELBOTerm("Hresp")
    if emptyPiFrac > 0:
        assert xSSslice.hasELBOTerm("HrespEmptyComp")

    # Prepare dict of info for debugging/inspection
    Info = dict()
    Info['Kfresh'] = xPiVec.size
    Info['xLPslice'] = xLPslice
    Info['xPiVec'] = xPiVec
    Info['emptyPi'] = emptyPi
    return xSSslice, Info

def restrictedLocalStep_HDPTopicModel(
        Dslice=None,
        curLPslice=None,
        ktarget=0,
        kabsorbList=None,
        xObsModel=None,
        xalphaPi=None,
        nUpdateSteps=3,
        doBuildOnInit=False,
        convThr=0.5,
        thetaEmptyComp=0.0,
        **kwargs):
    ''' Compute local parameters for HDPTopicModel via restricted local step.

    Returns
    -------
    xLPslice : dict with updated fields
        Fields with learned values
        * resp : N x Kfresh
        * DocTopicCount : nDoc x Kfresh
        * theta : nDoc x Kfresh
        * ElogPi : nDoc x Kfresh

        Fields copied directly from curLPslice
        * digammaSumTheta : 1D array, size nDoc
        * thetaRem : scalar
        * ElogPiRem : scalar
    '''
    if doBuildOnInit:
        xWholeSS = xInitSS.copy()
    Kfresh = xObsModel.K
    assert Kfresh == xalphaPi.size

    xLPslice = dict()
    # Default warm_start initialization for DocTopicCount
    # by copying the previous counts at all absorbing states
    if kabsorbList is None:
        xLPslice['DocTopicCount'] = np.zeros((Dslice.nDoc, Kfresh))
        xLPslice['resp'] = np.zeros((
            curLPslice['resp'].shape[0], Kfresh))
    else:
        # Initialize DocTopicCounts by copying those from absorbing states
        xLPslice['DocTopicCount'] = \
            curLPslice['DocTopicCount'][:, kabsorbList].copy()
        # Initialize resp by copying existing resp for absorbing state
        # Note: this is NOT consistent with some docs in DocTopicCount
        # but that will get fixed by restricted step
        xLPslice['resp'] = \
            curLPslice['resp'][:, kabsorbList].copy()

    xLPslice['theta'] = \
        xLPslice['DocTopicCount'] + xalphaPi[np.newaxis,:]

    xLPslice['_nIters'] = -1 * np.ones(Dslice.nDoc)
    xLPslice['_maxDiff'] = -1 * np.ones(Dslice.nDoc)

    for step in range(nUpdateSteps):
        # Compute conditional likelihoods for every data atom
        xLPslice = xObsModel.calc_local_params(Dslice, xLPslice)
        assert 'E_log_soft_ev' in xLPslice
        assert 'obsModelName' in xLPslice

        # Fill in these fields, one doc at a time
        for d in range(Dslice.nDoc):
            xLPslice = restrictedLocalStepForSingleDoc_HDPTopicModel(
                d=d,
                Dslice=Dslice,
                curLPslice=curLPslice,
                xLPslice=xLPslice,
                ktarget=ktarget,
                kabsorbList=kabsorbList,
                xalphaPi=xalphaPi,
                thetaEmptyComp=thetaEmptyComp,
                **kwargs)

        isLastStep = step == nUpdateSteps - 1
        if not isLastStep:
            xSS = xObsModel.calcSummaryStats(Dslice, None, xLPslice)
            # Increment
            if doBuildOnInit:
                xSS.setUIDs(xWholeSS.uids)
                xWholeSS += xSS
            else:
                xWholeSS = xSS
            # Global step
            xObsModel.update_global_params(xWholeSS)
            # Decrement stats
            if doBuildOnInit:
                xWholeSS -= xSS
            # Assess early stopping
            if step > 0:
                thr = np.sum(np.abs(prevCountVec - xSS.getCountVec()))
                if thr < convThr:
                    break
            prevCountVec = xSS.getCountVec()

    # Compute other LP quantities related to log prob (topic | doc)
    # and fill these into the expanded LP dict
    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xLPslice['digammaSumTheta'] = digammaSumTheta
    xLPslice['ElogPi'] = \
        digamma(xLPslice['theta']) - digammaSumTheta[:, np.newaxis]
    xLPslice['thetaRem'] = curLPslice['thetaRem'].copy()
    xLPslice['ElogPiRem'] = curLPslice['ElogPiRem'].copy()

    # Compute quantities related to leaving ktarget almost empty,
    # as we expand and transfer mass to other comps
    if thetaEmptyComp > 0:
        ElogPiEmptyComp = digamma(thetaEmptyComp) - digammaSumTheta
        xLPslice['thetaEmptyComp'] = thetaEmptyComp
        xLPslice['ElogPiEmptyComp'] = ElogPiEmptyComp

        # Compute quantities related to OrigComp, the original target cluster.
        # These need to be tracked and turned into relevant summaries
        # so that they can be used to created a valid proposal state "propSS"
        xLPslice['ElogPiOrigComp'] = curLPslice['ElogPi'][:, ktarget]
        xLPslice['gammalnThetaOrigComp'] = \
            np.sum(gammaln(curLPslice['theta'][:, ktarget]))
        slack = curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['theta'][:, ktarget]
        xLPslice['slackThetaOrigComp'] = np.sum(
            slack * curLPslice['ElogPi'][:, ktarget])

        if hasattr(Dslice, 'word_count') and \
                xLPslice['resp'].shape[0] == Dslice.word_count.size:
            xLPslice['HrespOrigComp'] = -1 * NumericUtil.calcRlogRdotv(
                curLPslice['resp'][:, ktarget], Dslice.word_count)
        else:
            xLPslice['HrespOrigComp'] = -1 * NumericUtil.calcRlogR(
                curLPslice['resp'][:, ktarget])
    return xLPslice

def restrictedLocalStepForSingleDoc_HDPTopicModel(
        d=0,
        Dslice=None,
        curLPslice=None,
        ktarget=0,
        kabsorbList=None,
        xalphaPi=None,
        xLPslice=None,
        LPkwargs=dict(),
        d_initTargetDocTopicCount="warm_start",
        thetaEmptyComp=0.0,
        **kwargs):
    ''' Perform restricted local step on one document.

    Returns
    -------
    xLPslice : dict with updated entries related to document d
        * resp
        * DocTopicCount
        * theta
    '''
    # Verify we have likelihoods
    assert 'E_log_soft_ev' in xLPslice
    assert 'obsModelName' in xLPslice
    obsModelName = xLPslice['obsModelName']
    # Verify prior
    Kfresh = xalphaPi.size
    assert xLPslice['E_log_soft_ev'].shape[1] == Kfresh

    if hasattr(Dslice, 'word_count') and obsModelName.count('Bern'):
        raise ValueError("TODO")
    start = Dslice.doc_range[d]
    stop = Dslice.doc_range[d+1]

    if kabsorbList is None:
        constrained_sumTheta_d = curLPslice['theta'][d,ktarget]
        # Establish the total mass we must reallocate
        constrained_sumResp_d = curLPslice['resp'][start:stop,ktarget]
    else:
        constrained_sumTheta_d = curLPslice['theta'][d,ktarget] + \
            np.sum(curLPslice['theta'][d, kabsorbList])
        # Establish the total mass we must reallocate
        constrained_sumResp_d = curLPslice['resp'][start:stop,ktarget] + \
            np.sum(curLPslice['resp'][start:stop, kabsorbList], axis=1)
    mask_d = np.arange(stop-start)

    if mask_d.size == 0:
        return xLPslice
    # Compute the conditional likelihood matrix for the target atoms
    # xCLik_d will always have an entry equal to one.
    if mask_d.size > 0:
        xCLik_d = xLPslice['E_log_soft_ev'][start + mask_d].copy()
        xCLik_d -= np.max(xCLik_d, axis=1)[:,np.newaxis]
        # Protect against underflow
        np.maximum(xCLik_d, -300, out=xCLik_d)
        np.exp(xCLik_d, out=xCLik_d)

    if hasattr(Dslice, 'word_count') and obsModelName.count('Mult'):
        wc_d = Dslice.word_count[start + mask_d]
        wc_d *= constrained_sumResp_d[mask_d]
    else:
        wc_d = constrained_sumResp_d[mask_d].copy()

    # Initialize doc-topic counts
    prevxDocTopicCount_d = -1 * np.ones(Kfresh)
    xDocTopicCount_d = xLPslice['DocTopicCount'][d, :].copy()

    if kabsorbList is None:
        doWarmStart = False # always cold start for birth move
    else:
        fracTargetMass_d = curLPslice['DocTopicCount'][d,ktarget] \
            / curLPslice['DocTopicCount'][d,:].sum()
        if fracTargetMass_d >= 0.05:
            doWarmStart = d_initTargetDocTopicCount.count("warm_start")
        else:
            doWarmStart = True

    if doWarmStart:
        # Initialize xDocTopicProb_d
        xDocTopicProb_d = xDocTopicCount_d + xalphaPi
        digamma(xDocTopicProb_d, out=xDocTopicProb_d)
        np.maximum(xDocTopicProb_d, -300, out=xDocTopicProb_d)
        np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
    else:
        # Cold start! xDocTopicProb_d[k] \approx alpha * Prob[k]
        xDocTopicProb_d = xalphaPi.copy()

    # Initialize xsumResp_d
    xsumResp_d = np.zeros(xCLik_d.shape[0])
    np.dot(xCLik_d, xDocTopicProb_d, out=xsumResp_d)

    maxDiff_d = -1
    for riter in range(LPkwargs['nCoordAscentItersLP']):
        # Update DocTopicCount_d
        np.dot(wc_d / xsumResp_d, xCLik_d,
               out=xDocTopicCount_d)
        xDocTopicCount_d *= xDocTopicProb_d

        # Update xDocTopicProb_d
        np.add(xDocTopicCount_d, xalphaPi,
            out=xDocTopicProb_d)
        digamma(xDocTopicProb_d, out=xDocTopicProb_d)
        # Protect against underflow
        np.maximum(xDocTopicProb_d, -300, out=xDocTopicProb_d)
        np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
        assert np.min(xDocTopicProb_d) > 0.0

        # Update xsumResp_d
        np.dot(xCLik_d, xDocTopicProb_d, out=xsumResp_d)

        # Check for convergence
        if riter % 5 == 0:
            maxDiff_d = np.max(np.abs(
                prevxDocTopicCount_d - xDocTopicCount_d))
            if maxDiff_d < LPkwargs['convThrLP']:
                break
        # Track previous DocTopicCount
        prevxDocTopicCount_d[:] = xDocTopicCount_d

    # Update xResp_d
    assert np.all(np.isfinite(xDocTopicCount_d))
    xResp_d = xCLik_d
    xResp_d *= xDocTopicProb_d[np.newaxis, :]
    xResp_d /= xsumResp_d[:, np.newaxis]
    # Here, sum of each row of xResp_d is equal to 1.0
    # Need to make sum of each row equal mass on target cluster
    xResp_d *= constrained_sumResp_d[mask_d,np.newaxis]
    np.maximum(xResp_d, 1e-100, out=xResp_d)

    # Right here, xResp_d and xDocTopicProb_d
    # are exactly equal to one fwd step from the current xDocTopicCount_d
    # So, we can use our short-cut ELBO calculation.
    if False:
        #curLPslice['DocTopicCount'][d, ktarget] > 10.0
        L_doc_theta = np.sum(gammaln(xDocTopicCount_d + xalphaPi)) \
            - np.inner(xDocTopicCount_d, np.log(xDocTopicProb_d))
        L_doc_resp = np.inner(wc_d, np.log(xsumResp_d))
        L_doc = L_doc_resp + L_doc_theta
        #print "d=%3d  L_d=% .4e" % (d, L_doc)
        #print " ".join(["%6.1f" % (x) for x in xDocTopicCount_d])
        #xLPslice['L_doc'] = L_doc

    # Pack up into final LP dict
    # Taking one forward step so xDocTopicCount_d is consistent with xResp_d
    xLPslice['resp'][start+mask_d] = xResp_d
    if hasattr(Dslice, 'word_count') and obsModelName.count('Mult'):
        xDocTopicCount_d = np.dot(Dslice.word_count[start+mask_d], xResp_d)
    else:
        xDocTopicCount_d = np.sum(xResp_d, axis=0)

    xLPslice['DocTopicCount'][d, :] = xDocTopicCount_d
    xLPslice['theta'][d, :] = xalphaPi + xDocTopicCount_d
    xLPslice['_nIters'][d] = riter
    xLPslice['_maxDiff'][d] = maxDiff_d

    # Final verifcation that output meets required constraints
    respOK = np.allclose(
        xLPslice['resp'][start:stop].sum(axis=1),
        constrained_sumResp_d,
        atol=0.0001,
        rtol=0)
    assert respOK
    thetaOK = np.allclose(
        xLPslice['theta'][d, :].sum() + thetaEmptyComp,
        constrained_sumTheta_d,
        atol=0.0001,
        rtol=0)
    assert thetaOK
    # That's all folks
    return xLPslice
