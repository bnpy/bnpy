import numpy as np


def refineProposedRespViaLocalGlobalStepsAndDeletes(
        Data_n, propLP_n, tempModel, tempSS,
        propK=None,
        origK=None,
        nRefineIters=3,
        verbose=0,
        **kwargs):
    ''' Improve proposed LP via standard updates.

    Args
    -------
    tempModel : HModel
        represents whole dataset seen thus far, EXCLUDING current sequence n
        not required to have any new states unique to sequence n
    tempSS : SuffStatBag
        represents whole dataset seen thus far, EXCLUDING current sequence n
        not required to have any new states unique to sequence n

    Returns
    -------
    propLP_n : dict with updated local params
        Num comps finalpropK will be between origK and propK
    tempModel : HModel
        Num comps will be equal to finalpropK
        represents whole dataset seen thus far, including current sequence n
    tempSS : SuffStatBag
        Num comps will be equal to finalpropK
        represents whole dataset seen thus far, including current sequence n

    '''
    assert propLP_n['resp'].shape[1] == propK

    # Initialize tempSS and tempModel
    # to be fully consistent with propLP_n
    propSS_n = tempModel.get_global_suff_stats(Data_n, propLP_n)
    assert propSS_n.K == propK

    if tempSS is None:
        tempSS = propSS_n.copy()
    else:
        Kextra = propK - tempSS.K
        if Kextra > 0:
            tempSS.insertEmptyComps(Kextra)
        tempSS += propSS_n
    tempModel.update_global_params(tempSS)

    # Refine via repeated local/global steps
    for step in range(nRefineIters):
        propLP_n = tempModel.calc_local_params(Data_n)
        tempSS -= propSS_n
        propSS_n = tempModel.get_global_suff_stats(Data_n, propLP_n)
        tempSS += propSS_n
        tempModel.update_global_params(tempSS)
    # Here, tempSS and tempModel are fully-consistent with propLP_n
    assert tempSS.K == propK
    assert tempModel.obsModel.K == propK

    # Perform "consistent" removal of empty components
    # This guarantees that tempModel and tempSS reflect
    # the whole dataset, including updated propLP_n/propSS_n
    propLP_n, propSS_n, tempModel, tempSS, Info = \
        deleteEmptyCompsAndKeepConsistentWithWholeDataset(
            Data_n, propSS_n, tempModel, tempSS,
            propLP_n=propLP_n,
            origK=origK,
            verbose=verbose,
            **kwargs)

    assert tempSS.N.sum() >= propSS_n.N.sum() - 1e-7
    return propLP_n, tempModel, tempSS, Info


def deleteEmptyCompsAndKeepConsistentWithWholeDataset(
        Data_n, propSS_n, tempModel, tempSS,
        propLP_n=None,
        origK=None,
        verbose=0,
        **kwargs):
    ''' Remove empty components and return consistent model/SS values.

    Args
    -------
    Data_n
    propSS_n
    tempModel
    tempSS
    Info

    Returns
    -------
    propLP_n
    propSS_n
    tempModel
    tempSS
    Info
    '''
    if propLP_n is None:
        # Do one first step
        propLP_n = tempModel.calc_local_params(Data_n, limitMemoryLP=1)
        tempSS -= propSS_n
        propSS_n = tempModel.get_global_suff_stats(Data_n, propLP_n)
        tempSS += propSS_n
        tempModel.update_global_params(tempSS)

    # Remove any empty extra components
    extraIDs_remaining = np.arange(origK, tempSS.K).tolist()
    nEmpty = np.sum(tempSS.N[origK:] <= 1)
    if verbose:
        print(extraIDs_remaining, '<< original extra ids')
    while nEmpty > 0 and tempSS.K > 1:
        # Loop thru each remaining extra comp created for current seq. n
        # Remove any such comps that are too small.
        L = len(extraIDs_remaining)
        for kLoc, kk in enumerate(reversed(np.arange(origK, tempSS.K))):
            if tempSS.N[kk] <= 1 and tempSS.K > 1:
                if verbose:
                    print('removing extra comp %d' % (kk))
                tempSS.removeComp(kk)
                propSS_n.removeComp(kk)
                extraIDs_remaining.pop(L - kLoc - 1)
        tempModel.update_global_params(tempSS)
        # At this point, tempSS and tempModel have <= propK comps

        # Do complete local/global update
        # to make all parameters consistent again.
        propLP_n = tempModel.calc_local_params(Data_n, limitMemoryLP=1)
        tempSS -= propSS_n
        propSS_n = tempModel.get_global_suff_stats(Data_n, propLP_n)
        tempSS += propSS_n
        nEmpty = np.sum(tempSS.N[origK:] <= 1)
        # ... end loop removing empties

    # Do final update to make tempModel consistent with tempSS
    tempModel.update_global_params(tempSS)

    if verbose:
        print(extraIDs_remaining, '<< remaining extra ids AFTER')

    # Store the original ids of remaining extra comps
    # which are useful for visualizing how this refinement
    # step has updated the initial proposal
    Info = dict(
        extraIDs_remaining=np.asarray(extraIDs_remaining),
    )

    return propLP_n, propSS_n, tempModel, tempSS, Info
