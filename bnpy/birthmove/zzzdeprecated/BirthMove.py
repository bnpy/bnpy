'''
'''
import numpy as np

import BirthCreate
import BirthRefine
import BirthCleanup
from BirthProposalError import BirthProposalError
import VizBirth
from BirthLogger import log, logPhase


def run_birth_move(bigModel, bigSS, freshData, Q=None, Plan=None, **kwargsIN):
    ''' Run birth move on provided target data, creating up to Kfresh new comps

        Returns
        -------
        bigmodel
        bigSS
        MoveInfo
    '''
    logPhase('Target Data')
    if 'ktarget' in Plan:
        ktarget = Plan['ktarget']
        if 'targetUID' in Plan:
            know = np.flatnonzero(bigSS.uIDs == Plan['targetUID'])
            if know.size == 1:
                sizeNow = bigSS.getCountVec()[know[0]]
            else:
                sizeNow = 0
            log('target comp = %d. Size now %d. Size at selection %d.'
                % (Plan['targetUID'], sizeNow, Plan['count']),
                'moreinfo')
        else:
            log('ktarget= %d.' % (ktarget), 'moreinfo')
    log(freshData.get_stats_summary(), 'debug')

    kwargs = dict(**kwargsIN)  # make local copy!
    origids = dict(bigModel=id(bigModel), bigSS=id(bigSS))

    try:
        if bigSS is None:
            msg = "SKIPPED. SS must be valid SuffStatBag, not None."
            raise BirthProposalError(msg)

        if bigSS.K + kwargs['Kfresh'] > kwargs['Kmax']:
            kwargs['Kfresh'] = kwargs['Kmax'] - bigSS.K

        if kwargs['Kfresh'] < 1:
            msg = "SKIPPED. Reached upper limit of Kmax=%d comps."
            msg = msg % (kwargs['Kmax'])
            raise BirthProposalError(msg)

        # Determine baseline ELBO
        if kwargs['birthVerifyELBOIncrease']:
            curbigModel = bigModel.copy()
            nStep = 3
            curfreshLP = None
            for step in range(nStep):
                doELBO = (step == nStep - 1)  # only on last step
                curfreshLP = curbigModel.calc_local_params(
                    freshData, curfreshLP, **kwargs)
                curfreshSS = curbigModel.get_global_suff_stats(
                    freshData, curfreshLP, doPrecompEntropy=doELBO)
                if not doELBO:  # all but the last step
                    curbigModel.update_global_params(bigSS + curfreshSS)
            curELBO = curbigModel.calc_evidence(SS=curfreshSS)

        # Create freshModel, freshSS, both with Kfresh comps
        #  freshSS has scale freshData
        #  freshModel has arbitrary scale
        freshModel, freshSS, freshInfo = \
            BirthCreate.create_model_with_new_comps(
                bigModel, bigSS, freshData, Q=Q,
                Plan=Plan, **kwargs)

        # Visualize, if desired
        if 'doVizBirth' in kwargs and kwargs['doVizBirth']:
            VizBirth.viz_birth_proposal(bigModel, freshModel, Plan,
                                        curELBO=None, propELBO=None, **kwargs)
            input('>>>')
            from matplotlib import pylab
            pylab.close('all')

        # Create xbigModel and xbigSS, with K + Kfresh comps
        # freshData can be assigned to any of the K+Kfresh comps
        # so, any of the K+Kfresh comps may be changed
        # but original comps won't lose influence of bigSS
        # * xbigSS has scale bigData + freshData
        # * xbigModel has scale bigData + freshData
        if kwargs['expandOrder'] == 'expandThenRefine':
            xbigModel, xbigSS, xfreshSS, xInfo = \
                BirthRefine.expand_then_refine(
                    freshModel, freshSS, freshData,
                    bigModel, bigSS, **kwargs)
        else:
            raise NotImplementedError('TODO')

        if kwargs['birthVerifyELBOIncrease']:
            logPhase('Evaluation')
            assert xfreshSS.hasELBOTerms()
            propELBO = xbigModel.calc_evidence(SS=xfreshSS)
            didPass, ELBOmsg = make_acceptance_decision(curELBO, propELBO)
            log(ELBOmsg)
        else:
            didPass = True
            ELBOmsg = ''
            propELBO = None  # needed for kwarg for viz_birth_proposal
            curELBO = None

        Kcur = bigSS.K
        Ktotal = xbigSS.K
        birthCompIDs = list(range(Kcur, Ktotal))

        # Reject. Abandon the move.
        if not didPass:
            msg = "BIRTH REJECTED. Did not explain target better than current."
            raise BirthProposalError(msg)

        assert xbigModel.obsModel.K == xbigSS.K
        # Create dict of info about this birth move
        msg = 'BIRTH ACCEPTED. %d fresh comps.' % (len(birthCompIDs))
        log(msg, 'info')

        MoveInfo = dict(didAddNew=True,
                        msg=msg,
                        AdjustInfo=xInfo['AInfo'], ReplaceInfo=xInfo['RInfo'],
                        modifiedCompIDs=[],
                        birthCompIDs=birthCompIDs,
                        Korig=bigSS.K,
                        )
        MoveInfo.update(xInfo)
        MoveInfo.update(freshInfo)
        assert not xbigSS.hasELBOTerms()
        assert not xbigSS.hasMergeTerms()
        xfreshSS.removeELBOTerms()
        if kwargs['birthRetainExtraMass']:
            MoveInfo['extraSS'] = xfreshSS
            MoveInfo['modifiedCompIDs'] = list(range(Ktotal))
        else:
            # Restore xbigSS to same scale as original "big" dataset
            xbigSS -= xfreshSS
            assert np.allclose(xbigSS.N.sum(), bigSS.N.sum())

        if bigSS.hasMergeTerms():
            MergeTerms = bigSS._MergeTerms.copy()
            MergeTerms.insertEmptyComps(Ktotal - Kcur)
            xbigSS.restoreMergeTerms(MergeTerms)
        if bigSS.hasELBOTerms():
            ELBOTerms = bigSS._ELBOTerms.copy()
            ELBOTerms.insertEmptyComps(Ktotal - Kcur)
            if xInfo['AInfo'] is not None:
                for key in xInfo['AInfo']:
                    if hasattr(ELBOTerms, key):
                        arr = getattr(
                            ELBOTerms,
                            key) + bigSS.nDoc * xInfo['AInfo'][key]
                        ELBOTerms.setField(key, arr, dims='K')
            if xInfo['RInfo'] is not None:
                for key in xInfo['RInfo']:
                    if hasattr(ELBOTerms, key):
                        ELBOTerms.setField(
                            key,
                            bigSS.nDoc *
                            xInfo['RInfo'][key],
                            dims=None)
            xbigSS.restoreELBOTerms(ELBOTerms)

        return xbigModel, xbigSS, MoveInfo
    except BirthProposalError as e:
        # We execute this code when birth fails for any reason, including:
        #  * user-specified Kmax limit reached
        #  * cleanup phase removed all new components

        # Verify guarantees that input model and input suff stats haven't
        # changed
        assert origids['bigModel'] == id(bigModel)
        assert origids['bigSS'] == id(bigSS)

        # Write reason for failure to log
        log(str(e), 'moreinfo')

        # Return failure info
        MoveInfo = dict(didAddNew=False,
                        msg=str(e),
                        modifiedCompIDs=[],
                        birthCompIDs=[])
        return bigModel, bigSS, MoveInfo


def make_acceptance_decision(curELBO, propELBO):
    # Sanity check
    # TODO: type check to avoid this on Gauss models
    if propELBO > 0 and curELBO < 0:
        didPass = False
        ELBOmsg = " %.5e propEv is INSANE!" % (propELBO)
    else:
        percDiff = (propELBO - curELBO) / np.abs(curELBO)
        didPass = propELBO > curELBO and percDiff > 0.0001
        ELBOmsg = " %.5e propEv \n %.5e curEv" % (propELBO, curELBO)
    return didPass, ELBOmsg
