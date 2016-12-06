import numpy as np
import os

from bnpy.util import NumericUtil
from bnpy.util.SparseRespStatsUtil import calcSparseRlogR

def summarizeRestrictedLocalStep_DPMixtureModel(
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
        doBuildOnInit=False,
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
    assert xUIDs is not None
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

    # Create probabilities for each of the Kfresh new clusters
    # by subdividing the target comp's original probabilities
    xPiVec, emptyPi = make_xPiVec_and_emptyPi(
        curModel=curModel, xInitSS=xInitSS,
        ktarget=ktarget, Kfresh=Kfresh, **kwargs)
    
    sumRespVec = curLPslice['resp'][:,ktarget]
    isExpansion = True
    if np.intersect1d(xUIDs, curSSwhole.uids).size > 0:
        isExpansion = False
        for uid in xUIDs:
            kk = curSSwhole.uid2k(uid)
            sumRespVec += curLPslice['resp'][:, kk]

    # Perform restricted inference!
    # xLPslice contains local params for all Kfresh expansion clusters
    xLPslice = restrictedLocalStep_DPMixtureModel(
        Dslice=Dslice,
        sumRespVec=sumRespVec,
        xObsModel=xObsModel,
        xPiVec=xPiVec,
        doBuildOnInit=doBuildOnInit,
        xInitSS=xInitSS,
        **kwargs)
    
    # Summarize this expanded local parameter pack
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        doPrecompEntropy=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(xUIDs)

    # Handle bookkeeping for original entropy term
    if isExpansion:
        if 'resp' in curLPslice:
            HrespOrigComp = -1 * NumericUtil.calcRlogR(
                curLPslice['resp'][:, ktarget])
        else:
            target_resp = curLPslice['spR'][:, ktarget].toarray()
            np.maximum(target_resp, 1e-100, out=target_resp)
            HrespOrigComp = -1 * NumericUtil.calcRlogR(target_resp)
        xSSslice.setELBOTerm('HrespEmptyComp', -1 * HrespOrigComp, dims=None)

    # If desired, add merge terms into the expanded summaries,
    if mUIDPairs is not None and len(mUIDPairs) > 0:
        # Check just to be safe
        for uidA, uidB in mUIDPairs:
            assert uidA != targetUID
            assert uidB != targetUID
        Mdict = curModel.allocModel.calcMergeTermsFromSeparateLP(
            Data=Dslice, 
            LPa=curLPslice, SSa=curSSwhole,
            LPb=xLPslice, SSb=xSSslice, 
            mUIDPairs=mUIDPairs)
        xSSslice.setMergeUIDPairs(mUIDPairs)
        for key, arr in Mdict.items():
            xSSslice.setMergeTerm(key, arr, dims='M')
    # Prepare dict of info for debugging/inspection
    Info = dict()
    Info['Kfresh'] = Kfresh
    Info['xPiVec'] = xPiVec
    Info['emptyPi'] = emptyPi
    Info['xInitSS'] = xInitSS
    Info['xLPslice'] = xLPslice
    return xSSslice, Info



def restrictedLocalStep_DPMixtureModel(
        Dslice=None,
        sumRespVec=None,
        LPkwargs=dict(),
        xObsModel=None,
        xPiVec=None,
        xInitSS=None,
        doBuildOnInit=False,
        nUpdateSteps=50,
        convThr=0.1,
        xPiPrior=1.0,
        logFunc=None,
        **kwargs):
    ''' Perform restricted local step on provided dataset.

    Returns
    -------
    xLPslice : dict with updated local parameters
        Obeys restriction that sum(resp, axis=1) equals sumRespVec
    '''
    if xInitSS is None:
        xWholeSS = None
    else:
        xWholeSS = xInitSS.copy()
    for step in range(nUpdateSteps):
        # Compute conditional likelihoods for every data atom
        xLPslice = xObsModel.calc_local_params(Dslice, **LPkwargs)
        assert 'E_log_soft_ev' in xLPslice
        xresp = xLPslice['E_log_soft_ev']
        xresp += np.log(xPiVec)[np.newaxis,:]
        # Calculate exp in numerically stable manner (first subtract the max)
        #  perform this in-place so no new allocations occur
        NumericUtil.inplaceExpAndNormalizeRows(xresp)
        # Enforce sum restriction
        xresp *= sumRespVec[:,np.newaxis]
        np.maximum(xresp, 1e-100, out=xresp)

        isLastStep = step == nUpdateSteps - 1
        if not isLastStep:
            xSS = xObsModel.calcSummaryStats(Dslice, None, dict(resp=xresp))
            # Increment
            if doBuildOnInit:
                xSS.setUIDs(xWholeSS.uids)
                xWholeSS += xSS
            else:
                xWholeSS = xSS
            # Global step
            xObsModel.update_global_params(xWholeSS)
            Nvec = xWholeSS.getCountVec()
            xPiVec = Nvec + xPiPrior
            # Decrement stats
            if doBuildOnInit:
                xWholeSS -= xSS
            # Assess early stopping
            if step > 0:
                thr = np.sum(np.abs(prevCountVec - xSS.getCountVec()))
                if thr < convThr:
                    break
            prevCountVec = xSS.getCountVec()
    if logFunc:
        msg = "restrictedLocalStep_DPMixtureModel"
        msg += " stopped after %3d of %d iters. thr=%.4f" % (
            step, nUpdateSteps, thr)
        logFunc(msg)
    xLPslice['resp'] = xresp
    del xLPslice['E_log_soft_ev'] # delete since we did inplace ops on it
    return xLPslice

"""
def restrictedLocalStep_DPMixtureModel(
        Dslice=None,
        curLPslice=None,
        ktarget=0,
        Kfresh=None,
        xPi=None,
        xLPslice=None, 
        LPkwargs=dict(),
        xObsModel=None,
        xPiVec=None,
        nUpdateSteps=50,
        xInitSS=None,
        **kwargs):
    ''' Perform restricted local step for dataset.

    Returns
    -------
    xLPslice : dict with updated entries related to document d
        * resp
        * DocTopicCount
        * theta
    '''
    if xInitSS is None:
        xWholeSS = None
    else:
        xWholeSS = xInitSS.copy()
    for step in range(nUpdateSteps):
        # Compute conditional likelihoods for every data atom
        xLPslice = xObsModel.calc_local_params(Dslice, **LPkwargs)
        assert 'E_log_soft_ev' in xLPslice
        xresp = xLPslice['E_log_soft_ev']
        xresp += np.log(xPiVec)[np.newaxis,:]
        # Calculate exp in numerically stable manner (first subtract the max)
        #  perform this in-place so no new allocations occur
        NumericUtil.inplaceExpAndNormalizeRows(xresp)
        if 'resp' in curLPslice:
            # Make each row sum to ktarget value
            xresp *= curLPslice['resp'][:, ktarget][:,np.newaxis]
        else:
            xresp *= curLPslice['spR'][:, ktarget].toarray()
        np.maximum(xresp, 1e-100, out=xresp)

        isLastStep = step == nUpdateSteps - 1
        if not isLastStep:
            xSS = xObsModel.calcSummaryStats(Dslice, None, dict(resp=xresp))

            if xInitSS is not None:
                xSS.setUIDs(xWholeSS.uids)
                xWholeSS += xSS
            else:
                xWholeSS = xSS
            xObsModel.update_global_params(xWholeSS)
            if xInitSS is not None:
                xWholeSS -= xSS
    xLPslice['resp'] = xresp
    del xLPslice['E_log_soft_ev'] # delete since we did inplace ops on it
    return xLPslice
"""

def makeExpansionSSFromZ_DPMixtureModel(
        Dslice=None, curModel=None, curLPslice=None,
        **kwargs):
    ''' Create expanded sufficient stats from Z assignments on target subset.

    Returns
    -------
    xSSslice : accounts for all data atoms in Dslice assigned to ktarget
    Info : dict
    '''
    xLPslice = makeExpansionLPFromZ_DPMixtureModel(
        Dslice=Dslice, curModel=curModel, curLPslice=curLPslice, **kwargs)
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        doPrecompEntropy=1, trackDocUsage=1, doTrackTruncationGrowth=1)

    if 'resp' in curLPslice:
        HrespOrigComp = -1 * NumericUtil.calcRlogR(
            curLPslice['resp'][:, kwargs['ktarget']])
    else:
        target_resp = curLPslice['spR'][:, kwargs['ktarget']].toarray()
        np.maximum(target_resp, 1e-100, out=target_resp)
        HrespOrigComp = -1 * NumericUtil.calcRlogR(target_resp)[0]
    xSSslice.setELBOTerm('HrespEmptyComp', -1 * HrespOrigComp, dims=None)

    xSSslice.setUIDs(kwargs['xInitSS'].uids.copy())
    Info = dict()
    Info['xLPslice'] = xLPslice
    return xSSslice, Info

def makeExpansionLPFromZ_DPMixtureModel(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        ktarget=None,
        xInitSS=None,
        targetZ=None,
        atomType=None,
        chosenDataIDs=None,
        **kwargs):
    ''' Create expanded local parameters from Z assignments on target subset.

    Returns
    -------
    xLP : dict with fields
        resp : N x Kfresh
    '''
    Kfresh = targetZ.max() + 1

    # Compute prior probability of each proposed comp
    xPiVec, emptyPi = make_xPiVec_and_emptyPi(
        curModel=curModel, ktarget=ktarget, Kfresh=Kfresh, 
        xInitSS=xInitSS, **kwargs)

    # Compute likelihood under each proposed comp
    xObsModel = curModel.obsModel.copy()
    xObsModel.update_global_params(xInitSS)

    xLPslice = xObsModel.calc_local_params(Dslice)
    xresp = xLPslice['E_log_soft_ev']
    xresp += np.log(xPiVec) # log prior probability
    xresp -= xresp.max(axis=1)[:,np.newaxis]
    np.exp(xresp, out=xresp)
    # Initialize xresp so each atom is normalized
    # This is the "default", for non-target atoms.
    xresp /= xresp.sum(axis=1)[:,np.newaxis]
    # Now, replace all targeted atoms with an all-or-nothing assignment
    if atomType == 'doc' and curModel.getAllocModelName().count('HDP'):
        for pos, d in enumerate(chosenDataIDs):
            start = Dslice.doc_range[d]
            stop = Dslice.doc_range[d+1]
            xresp[start:stop, :] = 1e-100
            xresp[start:stop, targetZ[pos]] = 1.0
    else:        
        for pos, n in enumerate(chosenDataIDs):
            xresp[n, :] = 1e-100
            xresp[n, targetZ[pos]] = 1.0
    assert np.allclose(1.0, xresp.sum(axis=1))


    if 'spR' in curLPslice:
        xresp *= curLPslice['spR'][:,ktarget].toarray()
    else:
        # Make resp consistent with ktarget comp
        xresp *= curLPslice['resp'][:, ktarget][:,np.newaxis]
    np.maximum(xresp, 1e-100, out=xresp)

    # Package up into xLPslice
    xLPslice['resp'] = xresp
    return xLPslice



def make_xPiVec_and_emptyPi(
        curModel=None, xInitSS=None,
        origPiVec=None,
        ktarget=0, Kfresh=0,
        emptyPiFrac=0.01,
        b_emptyPiFrac=None,
        b_method_xPi='uniform',
        **kwargs):
    ''' Create probabilities for newborn clusters and residual cluster.

    Args
    ----
    curModel : HModel, used for getting original cluster probability
    ktarget : int, identifies the target cluster in curModel

    Returns
    -------
    xPiVec : 1D array, size Kfresh
    emptyPi : scalar

    Post Condition
    --------------
    The original value of Pi[ktarget]  equals the sum of 
    xPiVec (a vector) and emptyPi (a scalar).

    Examples
    --------
    >>> origPiVec = np.asarray([0.5, 0.5])
    >>> xPiVec, emptyPi = make_xPiVec_and_emptyPi(origPiVec=origPiVec, 
    ...     ktarget=0, Kfresh=3, emptyPiFrac=0.25)
    >>> print emptyPi
    0.125
    >>> print xPiVec
    [ 0.125  0.125  0.125]
    '''
    if b_emptyPiFrac is not None:
        emptyPiFrac = b_emptyPiFrac
    if origPiVec is None:
        # Create temporary probabilities for each new cluster
        origPiVec = curModel.allocModel.get_active_comp_probs()
    else:
        origPiVec = np.asarray(origPiVec, dtype=np.float64)
    assert origPiVec.size >= ktarget
    targetPi = origPiVec[ktarget]
    emptyPi = emptyPiFrac * targetPi
    if b_method_xPi == 'uniform':
        xPiVec = (1-emptyPiFrac) * targetPi * np.ones(Kfresh) / Kfresh
    elif b_method_xPi == 'normalizedCounts':
        pvec = xInitSS.getCountVec()
        pvec = pvec / pvec.sum()
        xPiVec = (1-emptyPiFrac) * targetPi * pvec
    elif b_method_xPi == 'avgPi':
        if xInitSS.hasSelectionTerm('SumPi'):
            pvec = xInitSS.getSelectionTerm('SumPi') / xInitSS.nDoc
        else:
            pvec = np.ones(Kfresh) #xInitSS.getCountVec()
        pvec = pvec / pvec.sum()
        xPiVec = (1-emptyPiFrac) * targetPi * pvec        
    else:
        raise ValueError("Unrecognized b_method_xPi: " + b_method_xPi)
    assert np.allclose(np.sum(xPiVec) + emptyPi, targetPi)
    return xPiVec, emptyPi
