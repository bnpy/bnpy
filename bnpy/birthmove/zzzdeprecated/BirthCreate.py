'''
BirthCreate.py

Logic for *creating* new components given
*  dataset (some subsample of full-dataset scale N)
*  existing model (with K comps, of scale N)
*  existing suff stats (with K comps, of scale N)

'''
import numpy as np

from BirthProposalError import BirthProposalError
import BirthCleanup
from BirthLogger import log, logPhase, logPosVector
from bnpy.mergemove import MergeMove, MergePlanner

fastParams = dict(nCoordAscentItersLP=1, convThrLP=0.001)


def create_model_with_new_comps(bigModel, bigSS, freshData, Q=None,
                                Plan=None, **kwargs):
    '''

      Returns
      -------
      freshModel : HModel with Kfresh components,
                     scale *may not* be consistent with target dataset
      freshSS : SuffStatBag with Kfresh components,
                     scale will be consistent with target dataset
    '''
    Info = dict()
    freshModel = bigModel.copy()

    if kwargs['creationRoutine'] == 'targetWordFreq':
        freshModel.set_global_params(beta=np.ones(1), K=1,
                                     topics=Plan['targetWordFreq'][
            np.newaxis,
            :],
            wordcountTotal=freshData.word_count.sum()
        )
    elif kwargs['creationRoutine'] == 'findmissingtopics':
        freshModel = create_new_model_findmissingtopics(
            freshModel, freshData,
            bigModel, **kwargs)
    elif kwargs['creationRoutine'] == 'xspectral':
        assert Q is not None
        freshModel = create_new_model_expandedspectral(
            freshModel, Q, freshData, bigModel, **kwargs)
    elif kwargs['creationRoutine'] == 'spectralOnTarget':
        freshModel = create_new_model_spectralOnTarget(
            freshModel, freshData, bigModel, **kwargs)
    else:
        freshModel.init_global_params(freshData,
                                      K=kwargs['Kfresh'],
                                      initname=kwargs['creationRoutine'],
                                      **kwargs)

    logPhase('Creation')
    log('CreationRoutine: ' + kwargs['creationRoutine'], 'debug')
    log('Kfresh=%d' % (freshModel.obsModel.K), 'debug')

    if not kwargs['creationDoUpdateFresh']:
        # Create freshSS that would produce (nearly) same freshModel.obsModel
        # after a call to update_global_params
        freshSS._Fields.setAllFieldsToZero()
        if hasattr(freshSS, 'WordCounts'):
            topics = freshSS.WordCounts
            priorvec = freshModel.obsModel.obsPrior.lamvec
            for k in range(freshSS.K):
                topics[k, :] = freshModel.obsModel.comp[k].lamvec - priorvec
            freshSS.setField('WordCounts', topics, dims=('K', 'D'))
        return freshModel, freshSS, Info

    # Record initial model for posterity
    if kwargs['birthDebug']:
        Info['freshModelInit'] = freshModel.copy()

    # Complete several iterations to improve this fresh proposal
    for step in range(kwargs['creationNumIters']):
        freshLP = freshModel.calc_local_params(freshData, **fastParams)
        freshSS = freshModel.get_global_suff_stats(freshData, freshLP)
        freshModel.update_global_params(freshSS)
        if step < 3 or (step + 1) % 10 == 0:
            logPosVector(
                freshSS.N, label='iter %3d' %
                (step + 1), level='debug')
        if step > 1:
            maxDiff = np.max(np.abs(freshSS.N - prevN))
            if maxDiff < 1.0:
                break
        prevN = freshSS.N.copy()

    logPosVector(freshSS.N, label='after creation', level='moreinfo')
    if kwargs['birthDebug']:
        Info['freshModelRefined'] = freshModel.copy()

    if kwargs['cleanupDeleteEmpty']:
        Kbefore = freshSS.K
        freshModel, freshSS = BirthCleanup.delete_empty_comps(
            freshData, freshModel, freshSS, Korig=0, **kwargs)
        freshLP = freshModel.calc_local_params(freshData)
        freshSS = freshModel.get_global_suff_stats(freshData, freshLP)
        freshModel.update_global_params(freshSS)
        if freshSS.K < Kbefore:
            msg = 'after remove empty (size < %d)' % (kwargs['cleanupMinSize'])
            logPosVector(freshSS.N, label=msg, level='moreinfo')

    if kwargs['cleanupDeleteToImproveFresh']:
        freshModel, freshSS, ELBO = BirthCleanup.delete_comps_to_improve_ELBO(
            freshData, freshModel, LP=freshLP)
        Info['evBound'] = ELBO
        if kwargs['birthDebug']:
            Info['freshModelPostDelete'] = freshModel.copy()

    elif kwargs['cleanupMergeToImproveFresh']:
        Korig = freshSS.K
        while freshSS.K > 1:
            mPairIDs, MM = MergePlanner.preselect_candidate_pairs(
                freshModel, freshSS,
                preselect_routine='wholeELBO',
                doLimitNumPairs=0,
                returnScoreMatrix=1,
                **kwargs)
            freshLP = freshModel.calc_local_params(freshData)
            freshSS = freshModel.get_global_suff_stats(freshData, freshLP,
                                                       doPrecompEntropy=1,
                                                       doPrecompMergeEntropy=1,
                                                       mPairIDs=mPairIDs)
            freshModel.update_global_params(freshSS)
            freshELBO = freshModel.calc_evidence(SS=freshSS)
            freshModel, freshSS, freshELBO, Info = \
                MergeMove.run_many_merge_moves(
                    freshModel, freshSS, freshELBO,
                    mPairIDs, M=MM,
                    isBirthCleanup=1,
                    logFunc=log)
            if len(Info['AcceptedPairs']) == 0:
                break
        if freshSS.K < Korig:
            msg = 'after merges'
            logPosVector(freshSS.N, label=msg, level='moreinfo')

    if freshSS.K < 2:
        msg = "BIRTH failed. Fresh proposal does not prefer multiple comps."
        raise BirthProposalError(msg)

    return freshModel, freshSS, Info

# Topic-model
# creation


def create_new_model_findmissingtopics(freshModel, freshData,
                                       bigModel, LP=None,
                                       MIN_CLUSTER_SIZE=3,
                                       seed=0,
                                       **kwargs):
    import KMeansRex

    Kfresh = kwargs['Kfresh']
    K = bigModel.obsModel.K

    if LP is None:
        LP = bigModel.calc_local_params(freshData)
    Prior = np.exp(LP['E_logPi'])
    Lik = bigModel.obsModel.getElogphiMatrix()
    Lik = np.exp(Lik - Lik.max(axis=1)[:, np.newaxis])

    DocWordFreq_model = np.dot(Prior, Lik)
    DocWordFreq_model /= DocWordFreq_model.sum(axis=1)[:, np.newaxis]

    DocWordFreq_emp = freshData.to_sparse_docword_matrix().toarray()
    DocWordFreq_emp /= 1e-9 + DocWordFreq_emp.sum(axis=1)[:, np.newaxis]

    DocWordFreq_missing = DocWordFreq_emp - DocWordFreq_model
    np.maximum(0, DocWordFreq_missing, out=DocWordFreq_missing)
    DocWordFreq_missing /= 1e-9 + \
        DocWordFreq_missing.sum(axis=1)[:, np.newaxis]

    WordFreq_ctrs, Z = KMeansRex.RunKMeans(DocWordFreq_missing, Kfresh,
                                           initname='plusplus',
                                           Niter=10, seed=seed)
    Nk, binedges = np.histogram(np.squeeze(Z), np.arange(-0.5, Kfresh))

    if np.any(Nk < MIN_CLUSTER_SIZE):
        WordFreq_ctrs = WordFreq_ctrs[Nk >= MIN_CLUSTER_SIZE]
        Kfresh = WordFreq_ctrs.shape[0]

    np.maximum(1e-8, WordFreq_ctrs, out=WordFreq_ctrs)
    WordFreq_ctrs /= WordFreq_ctrs.sum(axis=1)[:, np.newaxis]

    freshModel.set_global_params(beta=np.ones(Kfresh) / Kfresh, K=Kfresh,
                                 topics=WordFreq_ctrs,
                                 wordcountTotal=freshData.word_count.sum()
                                 )
    return freshModel


def create_new_model_expandedspectral(freshModel, Q, freshData, bigModel,
                                      **kwargs):
    K = bigModel.obsModel.K
    topics = np.zeros((K, Q.shape[1]))
    for k in range(K):
        topics[k, :] = bigModel.obsModel.comp[k].lamvec
        topics[k, :] = topics[k, :] / topics[k, :].sum()

    Kfresh = kwargs['Kfresh']
    bestRows = GSU.FindAnchorsForExpandedBasis(Q, topics, Kfresh)
    newTopics = Q[bestRows] / Q[bestRows].sum(axis=1)[:, np.newaxis]
    freshModel.set_global_params(beta=np.ones(Kfresh) / Kfresh, K=Kfresh,
                                 topics=newTopics,
                                 wordcountTotal=freshData.word_count.sum()
                                 )
    return freshModel


def create_new_model_spectralOnTarget(freshModel, freshData, bigModel,
                                      **kwargs):
    K = bigModel.obsModel.K
    topics = np.zeros((K, freshData.vocab_size))
    for k in range(K):
        topics[k, :] = bigModel.obsModel.comp[k].lamvec
        topics[k, :] = topics[k, :] / topics[k, :].sum()

    Kfresh = kwargs['Kfresh']
    Q = freshData.to_sparse_docword_matrix().toarray()
    bestRows = GSU.FindAnchorsForExpandedBasis(Q, topics, Kfresh)
    newTopics = Q[bestRows] / Q[bestRows].sum(axis=1)[:, np.newaxis]
    freshModel.set_global_params(beta=np.ones(Kfresh) / Kfresh, K=Kfresh,
                                 topics=newTopics,
                                 wordcountTotal=freshData.word_count.sum()
                                 )
    return freshModel
