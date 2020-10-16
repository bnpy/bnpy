import numpy as np
from scipy.special import gammaln, digamma

import bnpy

def evaluateDeleteMoveCandidate_LP(
        Data, curModel, 
        curLP=None,
        propLP=None):
    propModel = curModel.copy()
    curModel = curModel.copy()
    Korig = curModel.allocModel.K

    propSS = curModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
    propModel.update_global_params(propSS)

    mPairIDs=[]
    for k in range(Korig-1):
        mPairIDs.append((k, Korig-1+k))
    print(mPairIDs)
    
    curSS = curModel.get_global_suff_stats(
        Data, curLP, 
        doPrecompEntropy=1)
    curModel.update_global_params(curSS)
    curELBO = curModel.calc_evidence(SS=curSS)
    print(' current ELBO: %.5f' % (curELBO))

    propSS = propModel.get_global_suff_stats(
        Data, propLP, 
        doPrecompEntropy=1, doPrecompMergeEntropy=1, mPairIDs=mPairIDs)
    propModel.update_global_params(propSS)
    propELBO = propModel.calc_evidence(SS=propSS)
    print('expanded ELBO: %.5f' % (propELBO))
    
    finalModel, finalSS, finalELBO, Info = \
        bnpy.mergemove.MergeMove.run_many_merge_moves(
            propModel, propSS, propELBO, mPairIDs)
    finalELBO = finalModel.calc_evidence(SS=finalSS)
    print('   final ELBO: %.5f' % (finalELBO))
    
    return finalModel, dict(
        SS=finalSS,
        ELBO=finalELBO,
        MergeInfo=Info
        )

def makeDeleteMoveCandidate_LP(
        Data, curLP, curModel, 
        targetCompID=10,
        deleteStrategy='truelabels',
        minResp=0.001,
        **curLPkwargs):
    '''

    Returns
    -------
    propcurLP : dict of local params
        Replaces targetCompID with K "new" states,
        each one tracking exactly one existing state.
    '''

    curResp = curLP['resp']
    maxRespValBelowThr = curResp[curResp < minResp].max()
    assert maxRespValBelowThr < 1e-90

    Natom, Korig = curResp.shape
    remCompIDs = np.setdiff1d(np.arange(Korig), [targetCompID])
    relDocIDs = np.flatnonzero(
        curLP['DocTopicCount'][:, targetCompID] > minResp)
    propResp = 1e-100 * np.ones((Natom, 2*(Korig - 1)))
    propResp[:, :Korig-1] = curResp[:, remCompIDs]

    if deleteStrategy.count('truelabels'):
        relAtoms = curResp[:, targetCompID] > minResp

        reltrueResp = Data.TrueParams['resp'][relAtoms].copy()
        reltrueResp[reltrueResp < minResp] = 1e-100
        reltrueResp /= reltrueResp.sum(axis=1)[:,np.newaxis]

        propResp[relAtoms, Korig-1:] = \
            reltrueResp * curResp[relAtoms, targetCompID][:,np.newaxis]
        propcurLP = curModel.allocModel.initLPFromResp(
            Data, dict(resp=propResp))
        return propcurLP
    
    Lik = curLP['E_log_soft_ev'][:, remCompIDs].copy()

    # From-scratch strategy
    for d in relDocIDs:
        mask_d = np.arange(Data.doc_range[d],Data.doc_range[d+1])
        relAtomIDs_d = mask_d[
            curLP['resp'][mask_d, targetCompID] > minResp]
        fixedDocTopicCount_d = curLP['DocTopicCount'][d, remCompIDs]
        relLik_d = Lik[relAtomIDs_d, :]
        relwc_d = Data.word_count[relAtomIDs_d]
        
        targetsumResp_d = curLP['resp'][relAtomIDs_d, targetCompID] * relwc_d
        sumResp_d = np.zeros_like(targetsumResp_d)
        
        DocTopicCount_d = np.zeros_like(fixedDocTopicCount_d)
        DocTopicProb_d = np.zeros_like(DocTopicCount_d)
        sumalphaEbeta = curModel.allocModel.alpha_E_beta()[targetCompID]
        alphaEbeta = sumalphaEbeta * 1.0 / (Korig-1.0) * np.ones(Korig-1)
        for riter in range(10):
            np.add(DocTopicCount_d, alphaEbeta, out=DocTopicProb_d)
            digamma(DocTopicProb_d, out=DocTopicProb_d)
            DocTopicProb_d -= DocTopicProb_d.max()
            np.exp(DocTopicProb_d, out=DocTopicProb_d)
            
            # Update sumResp for all tokens in document
            np.dot(relLik_d, DocTopicProb_d, out=sumResp_d)

            # Update DocTopicCount_d: 1D array, shape K
            #     sum(DocTopicCount_d) equals Nd[targetCompID]
            np.dot(targetsumResp_d / sumResp_d, relLik_d, out=DocTopicCount_d)
            DocTopicCount_d *= DocTopicProb_d
            DocTopicCount_d += fixedDocTopicCount_d

        DocTopicCount_dj = curLP['DocTopicCount'][d, targetCompID]
        DocTopicCount_dnew = np.sum(DocTopicCount_d) - \
            fixedDocTopicCount_d.sum()
        assert np.allclose(DocTopicCount_dj, DocTopicCount_dnew,
                           rtol=0, atol=1e-6)

        # Create proposal resp for relevant atoms in this doc only
        propResp_d = relLik_d.copy()
        propResp_d *= DocTopicProb_d[np.newaxis, :]
        propResp_d /= sumResp_d[:, np.newaxis]
        propResp_d *= curLP['resp'][relAtomIDs_d, targetCompID][:,np.newaxis]

        for n in range(propResp_d.shape[0]):
            size_n = curLP['resp'][relAtomIDs_d[n], targetCompID]
            sizeOrder_n = np.argsort(propResp_d[n,:])
            for k, compID in enumerate(sizeOrder_n):
                if propResp_d[n, compID] > minResp:
                    break
                propResp_d[n, compID] = 1e-100
                biggerCompIDs = sizeOrder_n[k+1:]
                propResp_d[n, biggerCompIDs] /= \
                    propResp_d[n,biggerCompIDs].sum()
                propResp_d[n, biggerCompIDs] *= size_n

        # Fill in huge resp matrix with specific values
        propResp[relAtomIDs_d, Korig-1:] = propResp_d
        assert np.allclose(propResp.sum(axis=1), 1.0, rtol=0, atol=1e-8)

    propcurLP = curModel.allocModel.initLPFromResp(Data, dict(resp=propResp))
    return propcurLP

def makeLPWithMinNonzeroValFromLP(Data, hmodel, LP, minResp=0.001):
    ''' Create sparse-ified local parameters, where all resp > threshold

    Returns
    -------
    sparseLP : dict of local parameters
    '''
    respS = LP['resp'].copy()
    Natom, Korig = respS.shape
    for n in range(Natom):
        sizeOrder_n = np.argsort(respS[n,:])
        for posLoc, compID in enumerate(sizeOrder_n):
            if respS[n, compID] > minResp:
                break
            respS[n, compID] = 1e-100
            biggerCompIDs = sizeOrder_n[posLoc+1:]
            respS[n, biggerCompIDs] /= respS[n, biggerCompIDs].sum()
    sparseLP = hmodel.allocModel.initLPFromResp(Data, dict(resp=respS))
    sparseLP['E_log_soft_ev'] = LP['E_log_soft_ev'].copy()
    return sparseLP
