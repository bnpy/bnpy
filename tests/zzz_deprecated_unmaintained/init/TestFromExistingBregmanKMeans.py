

import numpy as np
import Symbols as S
import bnpy
from scipy.sparse import csr_matrix
from bnpy.init.FromExistingBregman import runKMeans_BregmanDiv_existing
from bnpy.mergemove.MPlanner import selectCandidateMergePairs
from bnpy.viz.PlotUtil import pylab

if __name__ == '__main__':
    Npersymbol = 500
    Kfresh = 10

    # Create training set
    Xlist = list()
    Zlist = list()
    for ii, patch_name in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        X_ND = S.generate_patches_for_symbol(patch_name, Npersymbol)
        Xlist.append(X_ND)
        Zlist.append(ii * np.ones(X_ND.shape[0], dtype=np.int32))
    X = np.vstack(Xlist)
    TrainData = bnpy.data.XData(X, TrueZ=np.hstack(Zlist))
    TrainData.name = 'SimpleSymbols'

    # Train model on this set
    trainedModel, RInfo = bnpy.run(
        TrainData, 'DPMixtureModel', 'ZeroMeanGauss', 'memoVB',
        initname='truelabels',
        nLap=50, nBatch=1, 
        moves='merge', m_startLap=5,
        ECovMat='eye', sF=0.01,
        gamma=10.0)
    Korig = trainedModel.obsModel.K

    # Obtain local params and suff stats for this trained model
    trainLP = trainedModel.calc_local_params(TrainData)
    trainSS = trainedModel.get_global_suff_stats(
        TrainData, trainLP, doPrecompEntropy=1)

    # Create test set, with some novel clusters and some old ones
    # Create training set
    Xlist = list()
    for patch_name in ['A', 'B', 'C', 'D',
                       'slash', 'horiz_half', 'vert_half', 'cross']:
        X_ND = S.generate_patches_for_symbol(patch_name, Npersymbol)
        Xlist.append(X_ND)
    X = np.vstack(Xlist)
    TestData = bnpy.data.XData(X)
    TestData.name = 'SimpleSymbols'

    # Run FromExistingBregman procedure on test set
    print("Expanding model!")
    print("Creating %d new clusters via Bregman k-means++" % (Kfresh))
    print("Then assigning all %d test items to closest cluster")
    print("using union of %d existing and %d new clusters" % (
        Korig, Kfresh))
    Z, Mu, Lscores = runKMeans_BregmanDiv_existing(
        TestData.X, Kfresh, trainedModel.obsModel,
        assert_monotonic=False,
        Niter=5, logFunc=print)
    testLP = dict(
        nnzPerRow=1,
        spR=csr_matrix(
            (np.ones(Z.size), Z, np.arange(0, Z.size+1, 1)),
            shape=(TestData.nObs, Z.max() + 1))
        )
    testSS = trainedModel.get_global_suff_stats(TestData, testLP, doPrecompEntropy=1)

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
        testLP = combinedModel.calc_local_params(TestData)
        testSS = combinedModel.get_global_suff_stats(
            TestData, testLP, doPrecompEntropy=1)

        print("VB refinement iter %d" % aiter)
        print("   orig counts: ", 
            ' '.join(['%9.2f' % x for x in testSS.N[:Korig]]))
        print("  fresh counts: ", 
            ' '.join(['%9.2f' % x for x in testSS.N[Korig:]]))
        
        testSS.setUIDs(trainSS.uids)

        combinedSS = trainSS + testSS
        combinedModel.update_global_params(combinedSS)

        if aiter > 2:
            # Consider possible merges
            MInfo = selectCandidateMergePairs(combinedModel, combinedSS, 
                m_maxNumPairsContainingComp=1,
                lapFrac=aiter)
            if len(MInfo['m_UIDPairs']) > 0:
                for (uidA, uidB) in MInfo['m_UIDPairs']:
                    combinedSS.mergeComps(uidA=uidA, uidB=uidB)
                    trainSS.mergeComps(uidA=uidA, uidB=uidB)
                combinedModel.update_global_params(combinedSS)

    print()
    print("Plotting final combined model!")
    print("Each plot shows 25 samples of image patches from that cluster")

    PRNG = np.random.RandomState(0)
    for k in range(trainSS.K):
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
                axList[r,c].imshow(X_k[ii].reshape((8,8)),
                    interpolation='nearest',
                    cmap='gray_r',
                    vmin=-0.1,
                    vmax=0.1)
                ii += 1
                axList[r,c].set_xticks([])
                axList[r,c].set_yticks([])

    pylab.show()
