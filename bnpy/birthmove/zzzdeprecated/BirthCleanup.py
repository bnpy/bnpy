import numpy as np

from BirthLogger import log, logPosVector, logPhase
from BirthProposalError import BirthProposalError


def delete_comps_to_improve_ELBO(Data, model,
                                 SS=None, LP=None, ELBO=None,
                                 **kwargs):
    ''' Attempts deleting components K, K-1, K-2, ... Korig,
         keeping (and building on) any proposals that improve the ELBO

       Returns
       ---------
        model : HModel with K'' comps
        SS : SuffStatBag with K'' comps
        ELBO : evidence lower bound for the returned model
    '''
    if LP is None:
        LP = model.calc_local_params(Data)
    if ELBO is None and (SS is None or not SS.hasELBOTerms()):
        SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)
        ELBO = model.calc_evidence(SS=SS)
    elif ELBO is None:
        ELBO = model.calc_evidence(SS=SS)

    K = model.obsModel.K
    if K == 1:
        return model, SS, ELBO

    for k in reversed(list(range(0, K))):
        rmodel, rSS, rLP = _make_del_candidate__viaLP(Data, model, LP, k)
        assert rSS.hasELBOTerms()
        rELBO = rmodel.calc_evidence(SS=rSS)

        # If ELBO has improved, set current model to delete component k
        if rELBO >= ELBO:
            model = rmodel
            SS = rSS
            LP = rLP
            ELBO = rELBO
            # print 'delete accepted.  %.4e > %.4e' % (rELBO, ELBO)

        if SS.K == 1:
            break
        # end loop over comps to delete

    # if SS.K == 1:
    #  msg = "BIRTH failed. Deleting all fresh comps improves ELBO."
    #  raise BirthProposalError(msg)
    return model, SS, ELBO


def _make_del_candidate__viaLP(Data, model, LP, k):
    ''' Construct candidate model with deleted comp k from local params.
    '''
    rLP = _delete_comps_from_LP(Data, model, LP, k)
    rSS = model.get_global_suff_stats(Data, rLP, doPrecompEntropy=True)

    rmodel = model.copy()
    rmodel.update_global_params(rSS)
    return rmodel, rSS, rLP


def _delete_comps_from_LP(Data, model, LP, k):
    ''' Construct local params dict with components removed.
    '''
    if hasattr(model.allocModel, 'delete_comps_from_local_params'):
        rLP = model.allocModel.delete_comps_from_local_params(Data, LP, k)
    else:
        makeLPfunc = construct_LP_with_comps_removed
        rLP = makeLPfunc(Data, model, compIDs=[k], LP=LP)
    return rLP


def delete_comps_from_expanded_model_to_improve_ELBO(Data,
                                                     xbigModel, xbigSS,
                                                     xfreshSS, xfreshLP=None,
                                                     Korig=0, **kwargs):
    ''' Attempts deleting components K, K-1, K-2, ... Korig,
         keeping (and building on) any proposals that improve the ELBO

       Returns
       ---------
        model : HModel with Knew comps
        SS : SuffStatBag with Knew comps
        ELBO : evidence lower bound for the returned model
    '''
    logPhase('Cleanup')

    K = xbigSS.K
    assert xbigSS.K == xfreshSS.K
    assert xbigModel.obsModel.K == K

    origIDs = list(range(0, K))
    if K == 1:
        return xbigModel, xbigSS, xfreshSS, origIDs

    xfreshELBO = xbigModel.calc_evidence(SS=xfreshSS)
    for k in reversed(list(range(Korig, K))):
        if kwargs['cleanupDeleteViaLP']:
            rbigModel, rbigSS, rfreshSS, rfreshELBO, rfreshLP = \
                _make_xcandidate_LP(
                    xbigModel, Data,
                    xbigSS, xfreshSS, xfreshLP,
                    k, **kwargs)
        else:
            rbigModel, rbigSS, rfreshSS, rfreshELBO = _make_xcandidate(
                xbigModel, Data,
                xbigSS, xfreshSS,
                k)
        # If ELBO has improved, set current model to delete component k
        didAccept = False
        if rfreshELBO >= xfreshELBO:
            log('Deletion accepted. prop %.5e > cur %.5e' %
                (rfreshELBO, xfreshELBO))
            logPosVector(xfreshSS.N[Korig:])

            xbigSS = rbigSS
            xfreshSS = rfreshSS
            xbigModel = rbigModel
            xfreshELBO = rfreshELBO
            if kwargs['cleanupDeleteViaLP']:
                xfreshLP = rfreshLP
            didAccept = True
            del origIDs[k]

        if xfreshSS.K == 1:
            break
        # end loop over comps to delete

    if xbigSS.K == Korig and kwargs['cleanupRaiseErrorWhenAllDeleted']:
        log('FAILED. Deleting all new comps improves ELBO.')
        msg = "FAILED. After expansion, deleting all new comps improves ELBO."
        raise BirthProposalError(msg)
    return xbigModel, xbigSS, xfreshSS, xfreshELBO, origIDs


def _make_xcandidate(xbigModel, Data, xbigSS, xfreshSS, k):
    rbigModel = xbigModel.copy()
    rbigSS = xbigSS.copy()
    rfreshSS = xfreshSS.copy()

    rbigSS.removeComp(k)
    rfreshSS.removeComp(k)

    rSS = rbigSS + rfreshSS
    rbigModel.update_global_params(rSS)

    rLP = rbigModel.calc_local_params(Data)
    rfreshSS = rbigModel.get_global_suff_stats(
        Data,
        rLP,
        doPrecompEntropy=True)
    rfreshELBO = rbigModel.calc_evidence(SS=rfreshSS)

    return rbigModel, rbigSS, rfreshSS, rfreshELBO


def _make_xcandidate_LP(xbigModel, Data, xbigSS, xfreshSS, xfreshLP, k,
                        **kwargs):
    rfreshLP = _delete_comps_from_LP(Data, xbigModel, xfreshLP, k)
    rfreshSS = xbigModel.get_global_suff_stats(Data, rfreshLP,
                                               doPrecompEntropy=True)

    rbigModel = xbigModel.copy()
    rbigSS = xbigSS.copy()
    rbigSS.removeComp(
        rbigSS.K -
        1)  # just chop off the last one in stickbrk order

    qbigSS = rbigSS + rfreshSS
    rbigModel.update_global_params(qbigSS)

    # We might consider another pass to make sure the alloc params converge
    rbigModel.allocModel.update_global_params(qbigSS)

    if 'cleanupDeleteNumIters' in kwargs and kwargs['cleanupDeleteNumIters']:
        nIters = kwargs['cleanupDeleteNumIters']
        for trial in range(nIters):
            rfreshLP = rbigModel.calc_local_params(
                Data, rfreshLP, methodLP='memo',
                nCoordAscentItersLP=10)
            rfreshSS = rbigModel.get_global_suff_stats(Data, rfreshLP,
                                                       doPrecompEntropy=1)
            qbigSS = rbigSS + rfreshSS
            rbigModel.update_global_params(qbigSS)
            rfreshELBO = rbigModel.calc_evidence(SS=rfreshSS)
            log('%d  %.6e' % (trial, rfreshELBO))
    else:
        rfreshELBO = rbigModel.calc_evidence(SS=rfreshSS)
    return rbigModel, rbigSS, rfreshSS, rfreshELBO, rfreshLP


# delete empty comps
###########################################################
def delete_empty_comps(Data, model, SS=None,
                       Korig=0, **kwargs):
    ''' Removes any component K, K-1, K-2, ... Korig that is too small,
          as measured by SS.N[k]

        * does change allocmodel global params
        * does not alter any obsmodel global params for any comps.

       Returns
       ---------
        model : HModel, modified in-place to remove empty comps
        SS : SuffStatBag, modified in-place to remove empty comps
    '''

    if SS is None:
        LP = model.calc_local_params(Data)
        SS = model.get_global_suff_stats(Data, LP)

    K = SS.K
    for k in reversed(list(range(Korig, K))):
        if SS.N[k] < kwargs['cleanupMinSize']:
            if SS.K > 1:
                SS.removeComp(k)
            else:
                msg = 'FAILED. Created new comps below cleanupMinSize.'
                raise BirthProposalError(msg)

    if SS.K < model.allocModel.K:
        model.update_global_params(SS)
    return model, SS
