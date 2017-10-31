

import numpy as np
import itertools
import Symbols as S
import bnpy

from scipy.sparse import csr_matrix
from bnpy.init.FromExistingBregman import runKMeans_BregmanDiv_existing
from bnpy.mergemove.MPlanner import selectCandidateMergePairs
from bnpy.viz.PlotUtil import pylab
from bnpy.allocmodel.topics.LocalStepManyDocs import updateLPGivenDocTopicCount

np.set_printoptions(precision=4, suppress=1, linewidth=100)

def pprint_ELBO_dict_difference(
        ELBO_dict_1,
        ELBO_dict_2=None,
        keys=['Ldata', 'Lalloc', 'Lentropy', 'LcDtheta', 'Ltotal']):
    ''' Print easy-to-read info about which ELBO terms change 
    '''
    for key in keys:
        if ELBO_dict_2 is None:
            msg = '%8s   curL=% 6.3f' % (
                key,
                ELBO_dict_1[key],
                )
        else:
            msg = '%8s   curL=% 6.3f   propL=% 6.3f   diff=% 6.3f' % (
                key,
                ELBO_dict_1[key],
                ELBO_dict_2[key],
                ELBO_dict_2[key] - ELBO_dict_1[key],
                )
        print(msg)

if __name__ == '__main__':
    Npersymbol = 1000
    Ndoc = 10
    Kfresh = 10

    # Create training set
    # Each document will have exactly Npersymbol/Ndoc examples of each cluster
    # For exactly Npersymbol total examples of each cluster across the corpus
    Xlist = list()
    Zlist = list()
    doc_range = [0]
    PRNG = np.random.RandomState(0)
    for d in range(Ndoc):
        N_doc = 0
        for k, patch_name in enumerate(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
            N_d_k =  Npersymbol // Ndoc
            X_ND = S.generate_patches_for_symbol(patch_name, N_d_k)
            Xlist.append(X_ND)
            Zlist.append(k * np.ones(N_d_k, dtype=np.int32))
            N_doc += N_d_k
        doc_range.append(N_doc + doc_range[-1])
    X = np.vstack(Xlist)
    TrainData = bnpy.data.GroupXData(
        X,
        doc_range=doc_range,
        TrueZ=np.hstack(Zlist))
    TrainData.name = 'SimpleSymbols'

    # Train simple HDP model on this set
    trainedModel, RInfo = bnpy.run(
        TrainData, 'HDPTopicModel', 'ZeroMeanGauss', 'memoVB',
        initname='truelabels',
        nLap=50, nBatch=1,
        moves='',
        #moves='merge', m_startLap=5,
        ECovMat='eye', sF=0.01,
        gamma=10.0,
        alpha=0.5)
    Korig = trainedModel.obsModel.K

    # Obtain local params and suff stats for this trained model
    trainLP = trainedModel.calc_local_params(TrainData)
    trainSS = trainedModel.get_global_suff_stats(
        TrainData, trainLP, doPrecompEntropy=1, doTrackTruncationGrowth=1)

    # Create test set, with some novel clusters and some old ones
    Xlist = list()
    for patch_name in ['A', 'B', 'C', 'D',
                       'slash', 'horiz_half', 'vert_half', 'cross']:
        X_ND = S.generate_patches_for_symbol(patch_name, Npersymbol)
        Xlist.append(X_ND)
    X = np.vstack(Xlist)
    TestData = bnpy.data.GroupXData(X, doc_range=[0, len(X)])
    TestData.name = 'SimpleSymbols'

    # Run FromExistingBregman procedure on test set
    print("Expanding model!")
    print("Creating %d new clusters via Bregman k-means++" % (Kfresh))
    print("Then assigning all %d test items to closest cluster")
    print("using union of %d existing and %d new clusters" % (Korig, Kfresh))
    Z, Mu, Lscores = runKMeans_BregmanDiv_existing(
        TestData.X, Kfresh, trainedModel.obsModel,
        assert_monotonic=False,
        Niter=5, logFunc=print)
    Kall = Z.max() + 1
    Kfresh = Kall - Korig
    testLP = dict(
        nnzPerRow=1,
        spR=csr_matrix(
            (np.ones(Z.size), Z, np.arange(0, Z.size+1, 1)),
            shape=(TestData.nObs, Kall))
        )
    test_DocTopicCount = np.asarray(np.bincount(
        Z, minlength=Kall).reshape((1, Kall)), dtype=np.float64)
    testLP['DocTopicCount'] = test_DocTopicCount

    prev_pi0_rem = trainedModel.allocModel.E_beta_rem()
    new_pi0_rem = 0.01
    new_pi0_Kfresh = (1 - new_pi0_rem) / float(Kfresh) * np.ones(Kfresh)
    assert np.allclose(new_pi0_rem + new_pi0_Kfresh.sum(), 1.0)
    
    combined_pi0_Knew = np.hstack([
        trainedModel.allocModel.E_beta_active(),
        prev_pi0_rem * new_pi0_Kfresh,
        ])
    combined_pi0_rem = prev_pi0_rem * new_pi0_rem
    assert np.allclose(
        combined_pi0_Knew.sum() + combined_pi0_rem,
        1.0)

    # Now just multiply by alpha
    alpha = trainedModel.allocModel.alpha
    alphaPi0 = alpha * combined_pi0_Knew
    alphaPi0Rem = alpha * combined_pi0_rem
    assert np.allclose(
        alpha,
        alphaPi0.sum() + alphaPi0Rem)

    testLP = updateLPGivenDocTopicCount(
        testLP,
        test_DocTopicCount,
        alphaPi0,
        alphaPi0Rem)
    testSS = trainedModel.get_global_suff_stats(
        TestData,
        testLP,
        doPrecompEntropy=1,
        doTrackTruncationGrowth=1)

    print()
    print("Refining model!")
    print("Performing several full VB iterations")
    print("Merging any new clusters when VB objective approves")

    # Create a combined model for the train AND test set
    trainSS.insertEmptyComps(testSS.K - trainSS.K)
    combinedSS = trainSS + testSS
    combinedModel = trainedModel.copy()
    combinedModel.update_global_params(combinedSS)

    # Refine this combined model via several coord ascent passes thru TestData
    for aiter in range(10):
        if aiter > 2:
            doMergeThisIter = 1
            # Create list of all pairs of two uids that are both "fresh"
            m_UIDPairs_ff = [
                (uidA, uidB) for (uidA, uidB)
                in itertools.combinations(trainSS.uids[Korig:], 2)]
            # Create list of all pairs of two uids combining "orig" and "fresh"
            m_UIDPairs_of = [
                (uidA, uidB) for (uidA, uidB)
                in itertools.product(
                    trainSS.uids[:Korig],
                    trainSS.uids[Korig:])]
            # Create combined list
            # Be sure to try fresh/fresh pairs before orig/fresh pairs
            # m_UIDPairs = m_UIDPairs_of + m_UIDPairs_ff
            m_UIDPairs = m_UIDPairs_ff + m_UIDPairs_of
            m_IDPairs = [(trainSS.uid2k(uidA), trainSS.uid2k(uidB))
                for (uidA, uidB) in m_UIDPairs]
        else:
            doMergeThisIter = 0
            m_IDPairs = []
            m_UIDPairs = []
        
        assert combinedModel.obsModel.K == test_DocTopicCount.shape[1]
        init_test_LP = dict(DocTopicCount=test_DocTopicCount.copy())

        # Perform local step with "warm start"
        # using previous testLP's DocTopicCount attribute as initialization
        testLP = combinedModel.calc_local_params(
            TestData, init_test_LP,
            initDocTopicCountLP='memo',
            nCoordAscentItersLP=50,
            convThrLP=0.05)
        test_DocTopicCount = testLP['DocTopicCount'].copy()

        testSS = combinedModel.get_global_suff_stats(
            TestData, testLP,
            doPrecompEntropy=1, doTrackTruncationGrowth=1,
            doPrecompMergeEntropy=doMergeThisIter,
            mPairIDs=m_IDPairs)

        print("VB refinement iter %d" % aiter)
        print("   orig counts: ",
              ' '.join(['%9.2f' % x for x in testSS.N[:Korig]]))
        print("  fresh counts: ",
              ' '.join(['%9.2f' % x for x in testSS.N[Korig:]]))

        testSS.setUIDs(trainSS.uids)
        if len(m_UIDPairs) > 0:
            testSS.setMergeUIDPairs(m_UIDPairs)

        combinedSS = trainSS + testSS
        combinedModel.update_global_params(combinedSS)

        cur_ELBO = combinedModel.calc_evidence(SS=combinedSS)
        cur_ELBO_dict = combinedModel.calc_evidence(
            SS=combinedSS, todict=1)
        print("   ELBO % 8.5f" % cur_ELBO)
        pprint_ELBO_dict_difference(cur_ELBO_dict)

        if aiter > 2:
            # Track which uids were accepted
            acceptedUIDs = set()

            # Try merging each possible pair of uids
            for ii, (uidA, uidB) in enumerate(m_UIDPairs):
                if uidA in acceptedUIDs or uidB in acceptedUIDs:
                    # print(
                    #   'pair %2d %2d skipped. Comp accepted previously' % (
                    #    uidA, uidB))
                    continue

                kA = trainSS.uid2k(uidA)
                kB = trainSS.uid2k(uidB)

                # Remove empty training comp
                prop_trainSS = trainSS.copy()
                prop_trainSS.removeComp(uid=uidB)

                # Update proposed statistics the EASY BUT SLOW way
                # prop_testLP = \
                #    combinedModel.allocModel.applyHardMergePairToLP(
                #        testLP, kA, kB)
                # prop_testSS = combinedModel.get_global_suff_stats(
                #    TestData, prop_testLP,
                #    doPrecompEntropy=1,
                #    doTrackTruncationGrowth=1)
                # prop_testSS.setUIDs(prop_trainSS.uids)

                # Create proposed statistics the FASTER way
                # Uses precomputed fields within testSS._MergeTerms
                prop_testSS = testSS.copy()
                prop_testSS.mergeComps(uidA=uidA, uidB=uidB)
                
                # Create proposed model
                # Aggregating from both train and test
                prop_combinedSS = prop_trainSS + prop_testSS
                prop_combinedModel = combinedModel.copy()
                prop_combinedModel.update_global_params(prop_combinedSS)

                # If ELBO of proposed model improves, accept!
                prop_ELBO_dict = prop_combinedModel.calc_evidence(
                    SS=prop_combinedSS, todict=1)
                prop_ELBO = prop_combinedModel.calc_evidence(
                    SS=prop_combinedSS)

                if prop_ELBO > cur_ELBO:
                    # print('pair %2d %2d ACCEPTED!' % (uidA, uidB))
                    # print('cur gammalnTheta')
                    # print(testSS.getELBOTerm('gammalnTheta'))
                    # print('prop gammalnTheta')
                    # print(prop_testSS.getELBOTerm('gammalnTheta'))

                    # ACCEPT!
                    combinedModel = prop_combinedModel
                    trainSS = prop_trainSS
                    testSS = prop_testSS
                    test_DocTopicCount = \
                        prop_testSS.N[np.newaxis,:].copy()
                    acceptedUIDs.add(uidA)
                    acceptedUIDs.add(uidB)
                    print("   ELBO % 8.5f after accepted merge" % prop_ELBO)
                    pprint_ELBO_dict_difference(cur_ELBO_dict, prop_ELBO_dict)
                    cur_ELBO = prop_ELBO
                    cur_ELBO_dict = prop_ELBO_dict
                    
                else:
                    pass
                    # print('pair %2d %2d rejected' % (uidA, uidB))
                    # pprint_ELBO_dict_difference(cur_ELBO_dict, prop_ELBO_dict)
                    # print('cur gammalnTheta')
                    # print(testSS.getELBOTerm('gammalnTheta'))
                    # print('prop gammalnTheta')
                    # print(prop_testSS.getELBOTerm('gammalnTheta'))

    '''
    print()
    print("Plotting final combined model!")
    print("Each plot shows 25 samples of image patches from that cluster")

    PRNG = np.random.RandomState(0)
    for k in xrange(trainSS.K):
        Sigma_k = combinedModel.obsModel.get_covar_mat_for_comp(k)
        X_k = PRNG.multivariate_normal(
            np.zeros(64),
            Sigma_k,
            size=25)
        figH, axList = pylab.subplots(nrows=5, ncols=5)
        figH.canvas.set_window_title(
            'Cluster %d: Sample patches' % (k))

        ii = 0
        for r in range(5):
            for c in range(5):
                axList[r, c].imshow(
                    X_k[ii].reshape((8, 8)),
                    interpolation='nearest',
                    cmap='gray_r',
                    vmin=-0.1,
                    vmax=0.1)
                ii += 1
                axList[r, c].set_xticks([])
                axList[r, c].set_yticks([])

    pylab.show()
    '''
