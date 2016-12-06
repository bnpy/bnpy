import numpy as np

def expandLP_truelabels(
        Data_t, curLP_t, tmpModel, curSS_nott,
        **Plan):
    ''' Create single new state for all target data.

    Returns
    -------
    propLP_t : dict of local params, with K + 1 states
    xcurSS_nott : SuffStatBag
        first K states are equal to curSS_nott
        final few states are empty
    '''
    assert 'Z' in Data_t.TrueParams
    Z = Data_t.TrueParams['Z']
    uLabels = np.unique(Z)

    origK = curSS_nott.K
    propK = origK + len(uLabels)
    propResp = np.zeros((curLP_t['resp'].shape[0], propK))
    for uid, uval in enumerate(uLabels):
        mask_uid = Z == uval
        propResp[mask_uid, origK + uid] = 1.0
    propLP_t = dict(resp=propResp)
    if hasattr(tmpModel.allocModel, 'initLPFromResp'):
        propLP_t = tmpModel.allocModel.initLPFromResp(Data_t, propLP_t)
    # Make expanded xcurSS to match
    xcurSS_nott = curSS_nott.copy(includeELBOTerms=1, includeMergeTerms=0)
    xcurSS_nott.insertEmptyComps(propK - origK)
    return propLP_t, xcurSS_nott    

def expandLP_singleNewState(
        Data_t, curLP_t, tmpModel, curSS_nott,
        **Plan):
    ''' Create single new state for all target data.

    Returns
    -------
    propLP_t : dict of local params, with K + 1 states
    xcurSS_nott : SuffStatBag
        first K states are equal to curSS_nott
        final few states are empty
    '''
    xcurSS_nott = curSS_nott.copy(includeELBOTerms=1, includeMergeTerms=0)
    xcurSS_nott.insertEmptyComps(1)

    propK = curSS_nott.K + 1
    propResp = np.zeros((curLP_t['resp'].shape[0], propK))
    propResp[:, -1] = 1.0

    propLP_t = dict(resp=propResp)
    if hasattr(tmpModel.allocModel, 'initLPFromResp'):
        propLP_t = tmpModel.allocModel.initLPFromResp(Data_t, propLP_t)
    return propLP_t, xcurSS_nott

def expandLP_randomSplit(
        Data_t, curLP_t, tmpModel, curSS_nott,
        PRNG=np.random, **Plan):
    ''' Divide target data into two new states, completely at random.

    Returns
    -------
    propLP_t : dict of local params, with K + 2 states
    xcurSS_nott : SuffStatBag
        first K states are equal to curSS_nott
        final few states are empty
    '''
    Kfresh = 2
    xcurSS_nott = curSS_nott.copy(includeELBOTerms=1, includeMergeTerms=0)
    xcurSS_nott.insertEmptyComps(Kfresh)
    
    origK = curSS_nott.K
    propK = curSS_nott.K + Kfresh
    propResp = np.zeros((curLP_t['resp'].shape[0], propK))
    propResp[:, :origK] = curLP_t['resp']

    if 'btargetCompID' in Plan:
        atomids = np.flatnonzero(
            curLP_t['resp'][:, Plan['btargetCompID']] > 0.01)
    else:
        atomids = np.arange(propResp.shape[0])

    # randomly permute atomids
    PRNG.shuffle(atomids)
    if atomids.size > 20:
        Aids = atomids[:10]
        Bids = atomids[10:20]
    else:
        half = atomids.size / 2
        Aids = atomids[:half]
        Bids = atomids[half:]

    # Force all atomids to only be explained by new comps
    propResp[atomids, :] = 0.0
    propResp[Aids, -2] = 1.0
    propResp[Bids, -1] = 1.0

    propLP_t = dict(resp=propResp)
    if hasattr(tmpModel.allocModel, 'initLPFromResp'):
        propLP_t = tmpModel.allocModel.initLPFromResp(Data_t, propLP_t)

    propSS = tmpModel.get_global_suff_stats(Data_t, propLP_t)
    propSS += xcurSS_nott 
    tmpModel.update_global_params(propSS)

    propLP_t = tmpModel.calc_local_params(Data_t, propLP_t)
    return propLP_t, xcurSS_nott
