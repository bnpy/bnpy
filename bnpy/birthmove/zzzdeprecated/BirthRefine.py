import numpy as np

from BirthProposalError import BirthProposalError
from BirthLogger import log, logProbVector, logPosVector, logPhase
import BirthCleanup


def expand_then_refine(freshModel, freshSS, freshData,
                       bigModel, bigSS,
                       **kwargs):
    ''' Create expanded model with K + K' comps,
          then refine components K+1, K+2, ... K+K' via several VB iterations

        Guarantees that original comps of bigModel.obsModel are not altered.

        Returns
        -------
        xbigModel : HModel with K + Kfresh comps
                 * allocModel has scale bigSS + freshSS
                 * obsModel has scale bigSS + freshSS
        xbigSS : SuffStatBag with K + Kfresh comps
                  * has scale bigSS + freshSS
        xfreshSS : SuffStatBag with K + Kfresh comps
                  * has scale freshSS
        AdjustInfo : dict with adjustment factors
        ReplaceInfo : dict with replacement factors
    '''
    logPhase('Expansion')
    Korig = bigSS.K
    Info = dict()
    xbigModel = bigModel.copy()
    xbigSS = bigSS.copy(includeELBOTerms=False, includeMergeTerms=False)
    if kwargs['expandAdjustSuffStats'] \
            and hasattr(freshModel.allocModel, 'insertCompsIntoSuffStatBag'):
        xbigSS, AInfo, RInfo = xbigModel.allocModel.insertCompsIntoSuffStatBag(
            xbigSS, freshSS)
        log('Specialized, model-specific expansion')
        log('rho[K+1] ... rho[K+Knew]')
        logProbVector(xbigModel.allocModel.rho[Korig:])

    else:
        xbigSS.insertComps(freshSS)
        AInfo = None
        RInfo = None

    # Create expanded model, K + Kfresh comps
    Kx = xbigSS.K
    if xbigModel.allocModel.K < Kx:
        xbigModel.allocModel.update_global_params(xbigSS)
    if xbigModel.obsModel.K < Kx:
        xbigModel.obsModel.update_global_params(xbigSS)
    xbigSS.subtractSpecificComps(freshSS, list(range(bigSS.K, bigSS.K + freshSS.K)))
    if kwargs['birthDebug']:
        Info['xbigModelInit'] = xbigModel.copy()

    # Refine expanded model with VB iterations
    if kwargs['refineNumIters'] > 0:
        xbigModel, xfreshSS, xfreshLP, xInfo = \
            refine_expanded_model_with_VB_iters(
                xbigModel, freshData,
                xbigSS=xbigSS, Korig=bigSS.K, **kwargs)
    else:
        xfreshSS = xbigSS.copy()
        xfreshSS.setAllFieldsToZero()
        for key in list(xfreshSS._FieldDims.keys()):
            if xfreshSS._FieldDims[key] is None:
                continue
            arr = getattr(xfreshSS, key)
            arr[bigSS.K:] = getattr(freshSS, key)
        xfreshLP = None
        xInfo = dict(origIDs=list())

    if kwargs['birthDebug']:
        Info['xbigModelRefined'] = xbigModel.copy()
        Info['traceN'] = xInfo['traceN']
        Info['traceBeta'] = xInfo['traceBeta']
        Info['traceELBO'] = xInfo['traceELBO']

    AInfo = _delete_from_AInfo(AInfo, xInfo['origIDs'], Kx)
    if hasattr(xfreshSS, 'nDoc'):
        assert xbigSS.nDoc == bigSS.nDoc
        assert xfreshSS.nDoc == freshData.nDoc

    if kwargs['cleanupDeleteToImprove']:
        Kx = xbigSS.K
        xbigModel, xbigSS, xfreshSS, xfreshELBO, origIDs = \
            BirthCleanup.delete_comps_from_expanded_model_to_improve_ELBO(
                freshData, xbigModel,
                xbigSS, xfreshSS,
                Korig=bigSS.K, xfreshLP=xfreshLP, **kwargs)
        AInfo = _delete_from_AInfo(AInfo, origIDs, Kx)
        if kwargs['birthDebug']:
            Info['xbigModelPostDelete'] = xbigModel.copy()
            Info['ELBOPostDelete'] = xfreshELBO

    if hasattr(xfreshSS, 'nDoc'):
        assert xbigSS.nDoc == bigSS.nDoc
        assert xfreshSS.nDoc == freshData.nDoc
    xbigSS += xfreshSS
    Info['AInfo'] = AInfo
    Info['RInfo'] = RInfo
    return xbigModel, xbigSS, xfreshSS, Info


def refine_expanded_model_with_VB_iters(xbigModel, freshData,
                                        xbigSS=None, Korig=0, **kwargs):
    ''' Execute multiple local/global update steps for the current model

        Args
        --------
        xbigSS : SuffStatBag, with K + Kfresh comps,
                                   scale equal to bigData only

        Returns
        --------
        model : HModel, with K + Kfresh comps
                        scale equal to bigData + freshData
        freshSS : SuffStatBag, with K + Kfresh comps
                        scale equal to freshData
        freshLP : dict of local parameters for freshData


        Updates (in-place)
        ----------
        xbigSS : SuffStatBag, with K + Kfresh comps
                         scale with equal to bigData only
    '''
    logPhase('Refinement')

    xInfo = dict()
    origIDs = list(range(0, xbigSS.K))

    nIters = kwargs['refineNumIters']
    traceBeta = np.zeros((nIters, xbigSS.K))
    traceN = np.zeros((nIters, xbigSS.K))
    traceELBO = np.zeros(nIters)

    xfreshLP = None
    for riter in range(nIters):
        xfreshLP = xbigModel.calc_local_params(freshData, xfreshLP, **kwargs)
        xfreshSS = xbigModel.get_global_suff_stats(freshData, xfreshLP)

        traceN[riter, origIDs] = xfreshSS.N
        if kwargs['birthDebug']:
            traceBeta[
                riter,
                origIDs] = xbigModel.allocModel.get_active_comp_probs()
            traceELBO[riter] = xbigModel.calc_evidence(
                freshData,
                xfreshSS,
                xfreshLP)

        if riter < 3 or (riter + 1) % 5 == 0:
            logPosVector(traceN[riter, Korig:], label='iter %3d' % (riter + 1))

        # For all but last iteration, attempt removing empty topics
        if kwargs['cleanupDeleteEmpty'] and riter < kwargs[
                'refineNumIters'] - 1:
            for k in reversed(list(range(Korig, xfreshSS.K))):
                if xfreshSS.N[k] < kwargs['cleanupMinSize']:
                    xfreshSS.removeComp(k)
                    xbigSS.removeComp(xbigSS.K - 1)  # last in order!
                    del origIDs[k]

        if xfreshSS.K == Korig:
            msg = "BIRTH failed. After refining, no comps > cleanupMinSize."
            raise BirthProposalError(msg)

        xbigSS += xfreshSS
        xbigModel.allocModel.update_global_params(xbigSS)
        xbigModel.obsModel.update_global_params(xbigSS)
        xbigSS -= xfreshSS

    xfreshLP = xbigModel.calc_local_params(freshData, xfreshLP, **kwargs)
    xfreshSS = xbigModel.get_global_suff_stats(freshData, xfreshLP)
    log('Final Assignment Counts')
    logPosVector(xfreshSS.N[Korig:], label='final')

    if kwargs['birthDebug']:
        xInfo['traceBeta'] = traceBeta
        xInfo['traceN'] = traceN
        xInfo['traceELBO'] = traceELBO
    xInfo['origIDs'] = origIDs

    return xbigModel, xfreshSS, xfreshLP, xInfo


def _delete_from_AInfo(AInfo, origIDs, Kx):
    if AInfo is not None and len(origIDs) < Kx:
        for key in AInfo:
            # keep first in stick order!
            AInfo[key] = AInfo[key][:len(origIDs)]
    return AInfo
