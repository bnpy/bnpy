from builtins import *
import numpy as np
import os

from scipy.special import digamma, gammaln
from bnpy.allocmodel import make_xPiVec_and_emptyPi
from bnpy.util import NumericUtil

def summarizeRestrictedLocalStep_HDPTopicModel(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        curSSwhole=None,
        ktarget=None,
        targetUID=None,
        xUIDs=None,
        mUIDPairs=None,
        xObsModel=None,
        xInitSS=None,
        **kwargs):
    ''' Perform one restricted local step and summarize it.

    Returns
    -------
    xSSslice : SuffStatBag
    Info : dict with other information
    '''
    # Determine which uid to target
    if ktarget is None:
        assert targetUID is not None
        ktarget = curSSwhole.uid2k(targetUID)
    elif targetUID is None:
        assert ktarget is not None
        targetUID = curSSwhole.uids[ktarget]
    assert targetUID == curSSwhole.uids[ktarget]
    # Determine how many new uids to make
    Kfresh = len(xUIDs)
    # Verify provided summary states used to initialize clusters, if any.
    if xInitSS is not None:
        assert xInitSS.K == Kfresh
        xInitSS.setUIDs(xUIDs)
    # Create temporary observation model for each of Kfresh new clusters
    # If it doesn't exist already
    if xObsModel is None:
        xObsModel = curModel.obsModel.copy()
    if xInitSS is not None:
        xObsModel.update_global_params(xInitSS)
    assert xObsModel.K == Kfresh
    xPiVec, emptyPi = make_xPiVec_and_emptyPi(
        curModel=curModel, xInitSS=xInitSS,
        ktarget=ktarget, Kfresh=Kfresh, **kwargs)
    xalphaPi = curModel.allocModel.alpha * xPiVec
    thetaEmptyComp = curModel.allocModel.alpha * emptyPi

    # Perform restricted inference!
    # xLPslice contains local params for all Kfresh expansion clusters
    xLPslice = restrictedLocalStep_HDPTopicModel(
        Dslice=Dslice,
        curLPslice=curLPslice,
        ktarget=ktarget,
        xObsModel=xObsModel,
        xalphaPi=xalphaPi,
        thetaEmptyComp=thetaEmptyComp,
        **kwargs)
    assert "HrespOrigComp" in xLPslice

    # Summarize this expanded local parameter pack
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        trackDocUsage=1, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(xUIDs)
    assert xSSslice.hasELBOTerm("Hresp")
    if emptyPi > 0:
        assert xSSslice.hasELBOTerm("HrespEmptyComp")

    # If desired, add merge terms into the expanded summaries,
    if mUIDPairs is not None and len(mUIDPairs) > 0:
        Mdict = curModel.allocModel.calcMergeTermsFromSeparateLP(
            Data=Dslice,
            LPa=curLPslice, SSa=curSSwhole,
            LPb=xLPslice, SSb=xSSslice,
            mUIDPairs=mUIDPairs)
        xSSslice.setMergeUIDPairs(mUIDPairs)
        for key, arr in list(Mdict.items()):
            xSSslice.setMergeTerm(key, arr, dims='M')
    # Prepare dict of info for debugging/inspection
    Info = dict()
    Info['Kfresh'] = Kfresh
    Info['xInitSS'] = xInitSS
    Info['xLPslice'] = xLPslice
    Info['xPiVec'] = xPiVec
    Info['emptyPi'] = emptyPi
    return xSSslice, Info


def restrictedLocalStep_HDPTopicModel(
        Dslice=None,
        curLPslice=None,
        ktarget=0,
        xObsModel=None,
        xalphaPi=None,
        thetaEmptyComp=None,
        xInitLPslice=None,
        b_localStepSingleDoc='fast',
        **kwargs):
    '''

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

        * thetaEmptyComp
        * ElogPiEmptyComp
    '''
    Kfresh = xObsModel.K
    assert Kfresh == xalphaPi.size

    # Compute conditional likelihoods for every data atom
    xLPslice = xObsModel.calc_local_params(Dslice)
    assert 'E_log_soft_ev' in xLPslice

    # Initialize DocTopicCount and theta
    xLPslice['resp'] = xLPslice['E_log_soft_ev']
    xLPslice['DocTopicCount'] = np.zeros((Dslice.nDoc, Kfresh))
    xLPslice['theta'] = np.zeros((Dslice.nDoc, Kfresh))
    xLPslice['_nIters'] = -1 * np.ones(Dslice.nDoc)
    xLPslice['_maxDiff'] = -1 * np.ones(Dslice.nDoc)

    if b_localStepSingleDoc == 'fast':
        restrictedLocalStepForSingleDoc_Func = \
            restrictedLocalStepForSingleDoc_HDPTopicModel
    else:
        print('SLOW<<<!!')
        restrictedLocalStepForSingleDoc_Func = \
            restrictedLocalStepForSingleDoc_HDPTopicModel_SlowerButStable

    # Fill in these fields, one doc at a time
    for d in range(Dslice.nDoc):
        xLPslice = restrictedLocalStepForSingleDoc_Func(
            d=d,
            Dslice=Dslice,
            curLPslice=curLPslice,
            xLPslice=xLPslice,
            xInitLPslice=xInitLPslice,
            ktarget=ktarget,
            Kfresh=Kfresh,
            xalphaPi=xalphaPi,
            obsModelName=xObsModel.__class__.__name__,
            **kwargs)

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

    if isExpansion:
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
        d=0, Dslice=None, curLPslice=None,
        ktarget=0,
        Kfresh=None,
        xalphaPi=None,
        xLPslice=None,
        xInitLPslice=None,
        LPkwargs=dict(),
        obsModelName='Mult',
        **kwargs):
    ''' Perform restricted local step on one document.

    Returns
    -------
    xLPslice : dict with updated entries related to document d
        * resp
        * DocTopicCount
        * theta
    '''
    if hasattr(Dslice, 'word_count') and obsModelName.count('Bern'):
        start = d * Dslice.vocab_size
        stop = (d+1) * Dslice.vocab_size
        words_d = Dslice.word_id[Dslice.doc_range[d]:Dslice.doc_range[d+1]]
        mask_d = words_d[np.flatnonzero(
            curLPslice['resp'][start + words_d, ktarget] > 0.01)]
        # total of vocab_size atoms, subtract off present words
        lumpmask_d = np.setdiff1d(np.arange(Dslice.vocab_size), mask_d)

    else:
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]
        mask_d = np.flatnonzero(
            curLPslice['resp'][start:stop, ktarget] > 0.01)
        lumpmask_d = np.setdiff1d(np.arange(stop-start), mask_d)


    if hasattr(Dslice, 'word_count'):
        if obsModelName.count('Mult'):
            wc_d = Dslice.word_count[start + mask_d]
            wc_lump_d = Dslice.word_count[start + lumpmask_d]
        else:
            wc_d = 1.0
            wc_lump_d = 1.0
    else:
        wc_d = 1.0
        wc_lump_d = 1.0
    # Determine total mass assigned to each target atom,
    # We will learn how to redistribute this mass.
    targetsumResp_d = curLPslice['resp'][start + mask_d, ktarget] * wc_d
    # Compute total mass that will be dealt with as lump sum,
    # because it belongs to atoms that are too small to worry about.
    lumpMass_d = np.sum(
        curLPslice['resp'][start + lumpmask_d, ktarget] * wc_lump_d)
    # Allocate temporary memory for this document
    xsumResp_d = np.zeros_like(targetsumResp_d)
    xDocTopicCount_d = np.zeros(Kfresh)
    # Run coordinate ascent that alternatively updates
    # doc-topic counts and resp for document d
    if mask_d.size > 0:
        if xInitLPslice:
            xDocTopicCount_d[:] = xInitLPslice['DocTopicCount'][d, :]
        # Compute the conditional likelihood matrix for the target atoms
        # xCLik_d will always have an entry equal to one.
        assert 'E_log_soft_ev' in xLPslice
        xCLik_d = xLPslice['E_log_soft_ev'][start + mask_d]
        xCLik_d -= np.max(xCLik_d, axis=1)[:,np.newaxis]
        # Protect against underflow
        np.maximum(xCLik_d, -300, out=xCLik_d)
        np.exp(xCLik_d, out=xCLik_d)

        # Prepare doc-specific count vectors
        xDocTopicProb_d = np.zeros_like(xDocTopicCount_d)
        prevxDocTopicCount_d = -1 * np.ones(Kfresh)
        maxDiff_d = -1
        for riter in range(LPkwargs['nCoordAscentItersLP']):
            # xalphaEbeta_active_d potentially includes counts
            # for absorbing states from curLPslice_d
            np.add(xDocTopicCount_d, xalphaPi,
                out=xDocTopicProb_d)
            digamma(xDocTopicProb_d, out=xDocTopicProb_d)
            xDocTopicProb_d -= xDocTopicProb_d.max()
            # Protect against underflow
            np.maximum(xDocTopicProb_d, -300, out=xDocTopicProb_d)
            np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
            assert np.min(xDocTopicProb_d) > 0.0

            # Update sumResp for active tokens in document
            np.dot(xCLik_d, xDocTopicProb_d, out=xsumResp_d)

            # Update DocTopicCount_d: 1D array, shape K
            #     sum(DocTopicCount_d) equals Nd[ktarget]
            np.dot(targetsumResp_d / xsumResp_d, xCLik_d,
                   out=xDocTopicCount_d)
            xDocTopicCount_d *= xDocTopicProb_d

            if riter % 5 == 0:
                maxDiff_d = np.max(np.abs(
                    prevxDocTopicCount_d - xDocTopicCount_d))
                if maxDiff_d < LPkwargs['convThrLP']:
                    break
            prevxDocTopicCount_d[:] = xDocTopicCount_d

        xLPslice['_nIters'][d] = riter + 1
        xLPslice['_maxDiff'][d] = maxDiff_d
        # Make proposal resp for relevant atoms in current doc d
        if np.any(np.isnan(xDocTopicCount_d)):
            print('WHOA! NaN ALERT')
            # Edge case! Common only when deleting...
            # Recover from numerical issues in coord ascent
            # by falling back to likelihood only to make resp
            xResp_d = xCLik_d
            xResp_d /= xResp_d.sum(axis=1)[:,np.newaxis]
            np.dot(targetsumResp_d, xResp_d, out=xDocTopicCount_d)
        else:
            # Common case: Use valid result of coord ascent
            xResp_d = xCLik_d
            xResp_d *= xDocTopicProb_d[np.newaxis, :]
            xResp_d /= xsumResp_d[:, np.newaxis]

        # Here, sum of each row of xResp_d is equal to 1.0
        # Need to make sum of each row equal mass on target cluster
        xResp_d *= curLPslice['resp'][
            start + mask_d, ktarget][:, np.newaxis]
        np.maximum(xResp_d, 1e-100, out=xResp_d)
        assert np.allclose(
            xResp_d.sum(axis=1),
            curLPslice['resp'][start+mask_d, ktarget])
        xLPslice['resp'][start+mask_d] = xResp_d

    if lumpmask_d.size > 0:
        kmax = (xDocTopicCount_d + xalphaPi).argmax()
        xLPslice['resp'][start+lumpmask_d, :] = 1e-100
        xLPslice['resp'][start+lumpmask_d, kmax] = \
            curLPslice['resp'][start + lumpmask_d, ktarget]
        xDocTopicCount_d[kmax] += lumpMass_d

    # Fill in values in appropriate row of xDocTopicCount and xtheta
    xLPslice['DocTopicCount'][d, :] = xDocTopicCount_d
    xLPslice['theta'][d, :] = xalphaPi + xDocTopicCount_d
    assert np.allclose(xDocTopicCount_d.sum(),
                       curLPslice['DocTopicCount'][d, ktarget])
    assert np.allclose(
            xLPslice['resp'][start:stop, :].sum(axis=1),
            curLPslice['resp'][start:stop, ktarget])

    return xLPslice



def makeExpansionSSFromZ_HDPTopicModel(
        Dslice=None, curModel=None, curLPslice=None,
        **kwargs):
    ''' Create expanded sufficient stats from Z assignments on target subset.

    Returns
    -------
    xSSslice : accounts for all data atoms in Dslice assigned to ktarget
    Info : dict
    '''
    xLPslice = makeExpansionLPFromZ_HDPTopicModel(
        Dslice=Dslice, curModel=curModel, curLPslice=curLPslice, **kwargs)
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        doPrecompEntropy=1, trackDocUsage=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(kwargs['xInitSS'].uids.copy())
    Info = dict()
    Info['xLPslice'] = xLPslice
    return xSSslice, Info

def makeExpansionLPFromZ_HDPTopicModel(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        ktarget=None,
        xInitSS=None,
        targetZ=None,
        atomType=None,
        chosenDataIDs=None,
        emptyPiFrac=None,
        **kwargs):
    ''' Create expanded local parameters from Z assignments on target subset.

    Returns
    -------
    xLP : dict with fields
        resp : N x Kfresh
        DocTopicCount : D x Kfresh
        theta : D x Kfresh
        ElogPi : D x Kfresh
    '''
    Kfresh = targetZ.max() + 1
    N = curLPslice['resp'].shape[0]
    # Compute prior probability of each proposed comp
    xPiVec, emptyPi = make_xPiVec_and_emptyPi(
        curModel=curModel, ktarget=ktarget, Kfresh=Kfresh,
        xInitSS=xInitSS, **kwargs)
    xalphaPi = curModel.allocModel.alpha * xPiVec
    emptyalphaPi = curModel.allocModel.alpha * emptyPi

    # Compute likelihood under each proposed comp
    xObsModel = curModel.obsModel.copy()
    xObsModel.update_global_params(xInitSS)
    xLPslice = xObsModel.calc_local_params(Dslice)

    # Initialize xresp so each atom is normalized
    # This is the "default", for non-target atoms.
    xresp = xLPslice['E_log_soft_ev']
    xresp += np.log(xalphaPi) # log prior probability
    xresp -= xresp.max(axis=1)[:,np.newaxis]
    assert np.allclose(xresp.max(axis=1), 0.0)

    np.exp(xresp, out=xresp)
    xresp /= xresp.sum(axis=1)[:,np.newaxis]

    # Now, replace all targeted atoms with an all-or-nothing assignment
    if atomType == 'doc' and curModel.getAllocModelName().count('HDP'):
        if curModel.getObsModelName().count('Mult'):
            for pos, d in enumerate(chosenDataIDs):
                start = Dslice.doc_range[d]
                stop = Dslice.doc_range[d+1]
                xresp[start:stop, :] = 1e-100
                xresp[start:stop, targetZ[pos]] = 1.0
        elif curModel.getObsModelName().count('Bern'):
            # For all words in each targeted doc,
            # Assign them to the corresponding cluster in targetZ
            for pos, d in enumerate(chosenDataIDs):
                bstart = Dslice.vocab_size * d
                bstop = Dslice.vocab_size * (d+1)
                xresp[bstart:bstop, :] = 1e-100
                xresp[bstart:bstop, targetZ[pos]] = 1.0
                #words_d = Dslice.word_id[
                #    Dslice.doc_range[d]:Dslice.doc_range[d+1]]
                #xresp[bstart + words_d, :] = 1e-100
                #xresp[bstart + words_d, targetZ[pos]] = 1.0

    else:
        for pos, n in enumerate(chosenDataIDs):
            xresp[n, :] = 1e-100
            xresp[n, targetZ[pos]] = 1.0
    assert np.allclose(1.0, xresp.sum(axis=1))

    # Make resp consistent with ktarget comp
    xresp *= curLPslice['resp'][:, ktarget][:,np.newaxis]
    np.maximum(xresp, 1e-100, out=xresp)

    # Create xDocTopicCount
    xDocTopicCount = np.zeros((Dslice.nDoc, Kfresh))
    for d in range(Dslice.nDoc):
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]
        if hasattr(Dslice, 'word_id') and \
                curModel.getObsModelName().count('Mult'):
            xDocTopicCount[d] = np.dot(
                Dslice.word_count[start:stop],
                xresp[start:stop])
        elif hasattr(Dslice, 'word_id') and \
                curModel.getObsModelName().count('Bern'):
            bstart = d * Dslice.vocab_size
            bstop = (d+1) * Dslice.vocab_size
            xDocTopicCount[d] = np.sum(xresp[bstart:bstop], axis=0)
        else:
            xDocTopicCount[d] = np.sum(xresp[start:stop], axis=0)
    # Create xtheta
    xtheta = xDocTopicCount + xalphaPi[np.newaxis,:]

    # Package up into xLPslice
    xLPslice['resp'] = xresp
    xLPslice['DocTopicCount'] = xDocTopicCount
    xLPslice['theta'] = xtheta
    assert np.allclose(xDocTopicCount.sum(axis=1),
                       curLPslice['DocTopicCount'][:, ktarget])
    assert np.allclose(xtheta.sum(axis=1) + emptyalphaPi,
                       curLPslice['theta'][:, ktarget])

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
    if emptyalphaPi > 0:
        thetaEmptyComp = emptyalphaPi
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



def restrictedLocalStepForSingleDoc_HDPTopicModel_SlowerButStable(
        d=0, Dslice=None, curLPslice=None,
        ktarget=0,
        Kfresh=None,
        xalphaPi=None,
        xLPslice=None,
        xInitLPslice=None,
        LPkwargs=dict(),
        **kwargs):
    ''' Perform restricted local step on one document.

    Returns
    -------
    xLPslice : dict with updated entries related to document d
        * resp
        * DocTopicCount
        * theta
    '''
    start = Dslice.doc_range[d]
    stop = Dslice.doc_range[d+1]
    mask_d = np.flatnonzero(
        curLPslice['resp'][start:stop, ktarget] > 0.01)
    lumpmask_d = np.setdiff1d(np.arange(stop-start), mask_d)
    if hasattr(Dslice, 'word_count'):
        wc_d = Dslice.word_count[start + mask_d]
        wc_lump_d = Dslice.word_count[start + lumpmask_d]
    else:
        wc_d = 1.0
        wc_lump_d = 1.0
    # Determine total mass assigned to each target atom,
    # We will learn how to redistribute this mass.
    targetsumResp_d = curLPslice['resp'][start + mask_d, ktarget] * wc_d
    # Compute total mass that will be dealt with as lump sum,
    # because it belongs to atoms that are too small to worry about.
    lumpMass_d = np.sum(
        curLPslice['resp'][start + lumpmask_d, ktarget] * wc_lump_d)
    # Compute the conditional likelihood matrix for the target atoms
    # xCLik_d will always have an entry equal to one.
    assert 'E_log_soft_ev' in xLPslice
    xCLik_mask_d = xLPslice['E_log_soft_ev'][start + mask_d]
    xCLik_mask_d -= np.max(xCLik_mask_d, axis=1)[:,np.newaxis]
    # Allocate temporary memory for this document
    xsumResp_d = np.zeros_like(targetsumResp_d)
    xDocTopicCount_d = np.zeros(Kfresh)


    # Run coordinate ascent that alternatively updates
    # doc-topic counts and resp for document d
    if mask_d.size > 0:
        if xInitLPslice:
            xDocTopicCount_d[:] = xInitLPslice['DocTopicCount'][d, :]

        xresp_mask_d = np.zeros_like(xCLik_mask_d)

        xDocTopicProb_d = np.zeros_like(xDocTopicCount_d)
        prevxDocTopicCount_d = -1 * np.ones(Kfresh)
        for riter in range(LPkwargs['nCoordAscentItersLP']):
            # xalphaEbeta_active_d potentially includes counts
            # for absorbing states from curLPslice_d
            np.add(xDocTopicCount_d, xalphaPi,
                out=xDocTopicProb_d)
            digamma(xDocTopicProb_d, out=xDocTopicProb_d)
            xDocTopicProb_d -= xDocTopicProb_d.max()

            xresp_mask_d = xCLik_mask_d + xDocTopicProb_d[np.newaxis,:]
            xresp_mask_d -= xresp_mask_d.argmax(axis=1)[:,np.newaxis]
            np.exp(xresp_mask_d, out=xresp_mask_d)
            xresp_mask_d /= xresp_mask_d.sum(axis=1)[:,np.newaxis]
            xresp_mask_d *= \
                curLPslice['resp'][start + mask_d, ktarget][:,np.newaxis]

            # Compute doc topic count
            np.dot(wc_d, xresp_mask_d, out=xDocTopicCount_d)

            if riter % 5 == 0:
                maxDiff_d = np.max(np.abs(
                    prevxDocTopicCount_d - xDocTopicCount_d))
                if maxDiff_d < LPkwargs['convThrLP']:
                    break
            prevxDocTopicCount_d[:] = xDocTopicCount_d

        np.maximum(xresp_mask_d, 1e-100, out=xresp_mask_d)
        assert np.allclose(
            xresp_mask_d.sum(axis=1),
            curLPslice['resp'][start+mask_d, ktarget])
        xLPslice['resp'][start+mask_d] = xresp_mask_d

    if lumpmask_d.size > 0:
        kmax = (xDocTopicCount_d + xalphaPi).argmax()
        xLPslice['resp'][start+lumpmask_d, :] = 1e-100
        xLPslice['resp'][start+lumpmask_d, kmax] = \
            curLPslice['resp'][start + lumpmask_d, ktarget]
        xDocTopicCount_d[kmax] += lumpMass_d

    # Fill in values in appropriate row of xDocTopicCount and xtheta
    xLPslice['DocTopicCount'][d, :] = xDocTopicCount_d
    xLPslice['theta'][d, :] = xalphaPi + xDocTopicCount_d
    assert np.allclose(xDocTopicCount_d.sum(),
                       curLPslice['DocTopicCount'][d, ktarget])
    assert np.allclose(
            xLPslice['resp'][start:stop, :].sum(axis=1),
            curLPslice['resp'][start:stop, ktarget])
    return xLPslice

"""
def make_xalphaPi_and_emptyalphaPi(
        curModel=None, xInitSS=None,
        ktarget=0, Kfresh=0,
        emptyPiFrac=0.01, b_method_xPi='uniform', **kwargs):
    ''' Create probabilities for newborn clusters and residual cluster.

    Args
    ----
    curModel : HModel, used for getting original cluster probability
    ktarget : int, identifies the target cluster in curModel

    Returns
    -------
    xalphaPi : 1D array, size Kfresh
    emptyalphaPi : scalar

    Post Condition
    --------------
    Together, the total sum of xalphaPi (a vector) and emptyalphaPi (a scalar)
    equals the original value of alphaPi[ktarget].
    '''
    # Create temporary probabilities for each new cluster
    target_alphaPi = curModel.allocModel.alpha_E_beta()[ktarget]
    emptyalphaPi = emptyPiFrac * target_alphaPi
    if b_method_xPi == 'uniform':
        xalphaPi = (1-emptyPiFrac) * target_alphaPi * np.ones(Kfresh) / Kfresh
    elif b_method_xPi == 'normalized_counts':
        pvec = xInitSS.getCountVec()
        pvec = pvec / pvec.sum()
        xalphaPi = (1-emptyPiFrac) * target_alphaPi * pvec
    else:
        raise ValueError("Unrecognized b_method_xPi: " + b_method_xPi)
    assert np.allclose(np.sum(xalphaPi) + emptyalphaPi, target_alphaPi)
    return xalphaPi, emptyalphaPi
"""
