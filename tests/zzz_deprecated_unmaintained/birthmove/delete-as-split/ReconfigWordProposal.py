import numpy as np
from scipy.special import gammaln, digamma

import bnpy
from DeleteProposal import makeLPWithMinNonzeroValFromLP

np.set_printoptions(precision=3, suppress=1, linewidth=100)

def evaluateReconfigWordMoveCandidate_LP(
        Data, curModel, 
        curLP=None,
        propLP=None,
        targetCompID=0,
        destCompIDs=[1],
        targetWordIDs=None,
        **kwargs):
    propModel = curModel.copy()
    curModel = curModel.copy()
    Korig = curModel.allocModel.K

    # Evaluate current model
    curSS = curModel.get_global_suff_stats(
        Data, curLP, 
        doPrecompEntropy=1)
    curModel.update_global_params(curSS)
    curELBO = curModel.calc_evidence(SS=curSS)
    print(' current ELBO: %.5f' % (curELBO))

    # Visualize proposed expansion
    propSS = curModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
    propModel.update_global_params(propSS)

    mPairIDs = [(targetCompID, Korig)]
    if not destCompIDs[0] == targetCompID:
        destCompIDs = np.insert(destCompIDs, 0, targetCompID)
    for ctr, kk in enumerate(destCompIDs[1:]):
        mPairIDs.append((kk, Korig+ctr+1))
    print('Candidate merge pairs: ')
    print(mPairIDs)
    
    # Create full expansion (including merge terms)
    propSS = propModel.get_global_suff_stats(
        Data, propLP, 
        doPrecompEntropy=1, doPrecompMergeEntropy=1, mPairIDs=mPairIDs)
    propModel.update_global_params(propSS)
    propELBO = propModel.calc_evidence(SS=propSS)
    print('expanded ELBO: %.5f' % (propELBO))
    
    # Create final refined model after merging
    finalModel, finalSS, finalELBO, Info = \
        bnpy.mergemove.MergeMove.run_many_merge_moves(
            propModel, propSS, propELBO, mPairIDs)

    print('Accepted merge pairs: ')
    print(Info['AcceptedPairOrigIDs'])

    finalELBO = finalModel.calc_evidence(SS=finalSS)
    print('   final ELBO: %.5f' % (finalELBO))
    
    return finalModel, dict(
        SS=finalSS,
        ELBO=finalELBO,
        MergeInfo=Info
        )


def makeReconfigWordMoveCandidate_LP(
        Data, curLP, curModel, 
        targetWordIDs=[0,1,2,3,4,5],
        targetCompID=5,
        destCompIDs=[0],
        proposalName='truelabels',
        minResp=0.001,
        doShowViz=False,
        **curLPkwargs):
    ''' Create candidate local parameters dictionary for reconfig word move.

    Returns
    -------
    propLP : dict of local params
        Replaces targetCompID with K "new" states,
        each one tracking exactly one existing state.
    '''

    curResp = curLP['resp']
    maxRespValBelowThr = curResp[curResp < minResp].max()
    assert maxRespValBelowThr < 1e-90

    Natom, Korig = curResp.shape
    if destCompIDs[0] != targetCompID:
        destCompIDs = np.insert(destCompIDs, 0, targetCompID)
    Kprop = Korig + len(destCompIDs)

    propResp = 1e-100 * np.ones((Natom, Kprop))
    propResp[:, :Korig] = curResp
    if proposalName.count('truelabels'):
        # Identify atoms that match both target state and one-of target wordids
        relAtoms_byWords = np.zeros(Data.word_id.size, dtype=np.int32)
        for v in targetWordIDs:
            relAtoms_v = Data.word_id == v
            relAtoms_byWords = np.logical_or(relAtoms_byWords, relAtoms_v)
        relAtoms_byComp = curLP['resp'][:, targetCompID] > minResp
        relAtoms_twords = np.logical_and(relAtoms_byWords, relAtoms_byComp)
        relAtoms_nottwords = np.logical_and(
            relAtoms_byComp, 
            np.logical_not(relAtoms_byWords))
        
        # Keep non-target-word atoms assigned as they are
        propResp[relAtoms_nottwords, targetCompID] = \
            curResp[relAtoms_nottwords, targetCompID]
        propResp[relAtoms_twords, targetCompID] = 1e-100

        # Re-assign target-word atoms based on ground-truth labels
        reltrueResp = Data.TrueParams['resp'][relAtoms_twords].copy()
        reltrueResp = reltrueResp[:, destCompIDs]
        reltrueResp[reltrueResp < minResp] = 1e-100
        reltrueResp /= reltrueResp.sum(axis=1)[:,np.newaxis]
        reltrueResp *= curResp[relAtoms_twords, targetCompID][:, np.newaxis]
        assert np.allclose(curResp[relAtoms_twords, targetCompID],
                           reltrueResp.sum(axis=1))
        propResp[relAtoms_twords, Korig:] = reltrueResp

        assert np.allclose(1.0, propResp.sum(axis=1))
        propLP = curModel.allocModel.initLPFromResp(
            Data, dict(resp=propResp))
        return propLP

    relDocIDs = np.flatnonzero(
        curLP['DocTopicCount'][:, targetCompID] > minResp)
    relDocIDs_withwords = list()
    xlabels = list()
    for d in relDocIDs:
        word_id_d = Data.word_id[Data.doc_range[d]:Data.doc_range[d+1]]
        word_ct_d = Data.word_count[Data.doc_range[d]:Data.doc_range[d+1]]
        nTargetWords_d = 0
        for v in targetWordIDs:
            nTargetWords_d += np.sum(word_ct_d[word_id_d == v])
        if nTargetWords_d < 1:
            continue
        relDocIDs_withwords.append(d)

    # Show the raw doc words
    if doShowViz:
        xlabels = list()
        for d in relDocIDs_withwords:
            msg = 'N_d[trgt]=%5.0f  N_d[dest]=%5.0f' % (
                curLP['DocTopicCount'][d, targetCompID],
                curLP['DocTopicCount'][d, destCompIDs[1]],
                )
            xlabels.append(msg)
        bnpy.viz.BarsViz.plotExampleBarsDocs(
            Data, docIDsToPlot=relDocIDs_withwords[:16],
            xlabels=xlabels[:16],
            doShowNow=1)

    WType_x_Atom = Data.getTokenTypeCountMatrix()[targetWordIDs, :]
    for d in relDocIDs_withwords:
        # Find atoms in this doc that 
        # (1) use target words, and
        # (2) are assigned to target comp
        relAtomIDs_d = np.arange(Data.doc_range[d],Data.doc_range[d+1])
        relAtomIDs_dk = relAtomIDs_d[
            curLP['resp'][relAtomIDs_d, targetCompID] > minResp]
        relAtomIDs_dkv = relAtomIDs_dk[
            WType_x_Atom[:, relAtomIDs_dk].sum(axis=0) > 0]

        destRatio = curLP["DocTopicCount"][d, destCompIDs]
        destRatio /= destRatio.sum()

        destResp = destRatio * \
            curLP["resp"][relAtomIDs_dkv, targetCompID][:,np.newaxis]

        # Reassign curResp mass for target words using doc-topic counts alone
        propResp[relAtomIDs_dkv, targetCompID] = 1e-100
        propResp[relAtomIDs_dkv, Korig:] = destResp

        assert np.allclose(propResp.sum(axis=1), 1.0)


    propLP = curModel.allocModel.initLPFromResp(Data, dict(resp=propResp))
    return propLP


def makeReconfigWordPlan(Data, curSS, **kwargs):
    ''' Make plan for targeted proposal to reconfigure word assignments.

    Compare each pair j,k of existing topics, looking for
    a set of words that appears in topic j but not in topic k,
    yet has high co-occurance with words appearing in topic k but not j.
    The big idea is that these words may need to transfer some mass from j to k.    

    TODO
    ----
    Go beyond proposals that have just a pair of topics interacting.
    
    Returns
    -------
    Plan : dict with fields
    * targetCompID
    * targetWordsID
    * destCompIDs
    '''
    QMat = Data.getWordTypeCooccurMatrix()
    usedWordsByTopic = [None for k in range(curSS.K)]
    for k in range(curSS.K):
        usedWordsByTopic[k] = np.flatnonzero(curSS.WordCounts[k, :] > 10)

    bestScore = 0
    targetWordIDs = None
    destCompIDs = None
    for k in range(curSS.K):
        for kdest in range(curSS.K):
            if k == kdest:
                continue
            in_k_not_kdest = np.setdiff1d(usedWordsByTopic[k],
                                          usedWordsByTopic[kdest])
            in_kdest_not_k = np.setdiff1d(usedWordsByTopic[kdest],
                                          usedWordsByTopic[k])
            ScoreByOnTopicWord_k = \
                QMat[in_k_not_kdest, :][:, in_kdest_not_k].sum(axis=1)
            # Use all words that score more than twice median
            medianScore = np.median(ScoreByOnTopicWord_k)
            targetWordIDs_k = np.flatnonzero(
                ScoreByOnTopicWord_k > 2 * medianScore)
            curScore = ScoreByOnTopicWord_k[targetWordIDs_k].sum()
            # Record the total score for this combo, and the targetWordIDs
            if curScore > bestScore:
                bestScore = curScore
                targetWordIDs = in_k_not_kdest[targetWordIDs_k]
                destCompIDs = [k, kdest]
    return dict(
        targetWordIDs=targetWordIDs,
        destCompIDs=destCompIDs,
        targetCompID=destCompIDs[0],
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--initDropWordIDs', type=str, default='0,1,2,3,4,5')
    parser.add_argument('--initTargetCompID', type=int, default=0)
    parser.add_argument('--destCompIDs', type=str, default='5,0')
    parser.add_argument('--doPlan', type=int, default=0)
    args = parser.parse_args()

    import BarsK10V900
    Data = BarsK10V900.get_data(nDocTotal=300, nWordsPerDoc=200)

    LPkwargs = dict(
        nCoordAscentItersLP=50,
        convThrLP=0.001,
        restartLP=1,
        )
    curModel, Info = bnpy.run(
        Data, 'HDPTopicModel', 'Mult', 'moVB',
        lam=0.1, alpha=0.5, gamma=10,
        nLap=10, nBatch=2, printEvery=25,
        initname='truelabelsdropwords', 
        initTargetCompID=args.initTargetCompID,
        initDropWordIDs=args.initDropWordIDs,
        **LPkwargs)
    curLP = curModel.calc_local_params(Data, **LPkwargs)
    curLP = makeLPWithMinNonzeroValFromLP(Data, curModel, curLP)
    curSS = curModel.get_global_suff_stats(
        Data, curLP, doPrecompEntropy=1)

    if args.doPlan:
        Plan = makeReconfigWordPlan(Data, curSS)
    else:
        initDropWordIDs = [int(k) for k in args.initDropWordIDs.split(',')]
        destCompIDs = [int(k) for k in args.destCompIDs.split(',')]
        Plan = dict(
            targetWordIDs=initDropWordIDs,
            destCompIDs=destCompIDs,
            targetCompID=destCompIDs[0],
            )            

    print('Model is STUCK!')
    print('Ideal knowledge of missing words:', args.initDropWordIDs)
    print('Planned target words: ', Plan['targetWordIDs'])

    print('curSS counts of target words by topic')
    for k in Plan['destCompIDs']:
        print('topic %d: %s' % (
            k, str(curSS.WordCounts[k, Plan['targetWordIDs']])))

    print('')
    print('Proposing a candidate set of local parameters...')
    propLP_true = makeReconfigWordMoveCandidate_LP(
        Data, curLP, curModel, 
        proposalName='truelabels', **Plan)
    propModel_true, Result = evaluateReconfigWordMoveCandidate_LP(
        Data, curModel, curLP=curLP, propLP=propLP_true,
        proposalName='truelabels', **Plan)
    print('')
    print('Proposing a candidate set of local parameters...')
    propLP_scratch = makeReconfigWordMoveCandidate_LP(
        Data, curLP, curModel, 
        proposalName='fromscratch', **Plan)
    propModel_scratch, Result = evaluateReconfigWordMoveCandidate_LP(
        Data, curModel, curLP=curLP, propLP=propLP_scratch,
        proposalName='fromscratch', **Plan)

    propSS = Result['SS']
    for k in Plan['destCompIDs']:
        print('Topic %d: counts of target words' % (k))
        print(propSS.WordCounts[k, Plan['targetWordIDs']])

