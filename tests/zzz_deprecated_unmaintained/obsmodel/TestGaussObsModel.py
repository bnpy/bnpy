'''
'''
import numpy as np
import unittest
import copy

import bnpy


class TestHardMerge(unittest.TestCase):

    def setUp(self, K=5, D=2, N=50):
        PRNG = np.random.RandomState(0)

        PriorSpec = dict(alpha0=5, ECovMat='eye', sF=1.337, nu=0)
        oModel = bnpy.obsmodel.GaussObsModel('VB', D=2, **PriorSpec)

        Data = bnpy.data.XData(PRNG.randn(N, D))
        resp = PRNG.rand(N, K)
        resp /= resp.sum(axis=1)[:, np.newaxis]
        LP = dict(resp=resp)

        SS = oModel.get_global_suff_stats(Data, None, LP)
        oModel.update_global_params(SS)
        self.beforeModel = oModel
        self.beforeSS = SS
        self.beforeK = K

    def test_calcHardMergeGap(self):
        print('')
        beforeELBO = self.beforeModel.calcELBO_Memoized(self.beforeSS)

        afterModel = copy.deepcopy(self.beforeModel)
        GapMat = self.beforeModel.calcHardMergeGap_AllPairs(self.beforeSS)
        for kA in range(self.beforeK):
            for kB in range(kA + 1, self.beforeK):
                print('%d, %d' % (kA, kB))
                afterSS = self.beforeSS.copy()
                afterSS.mergeComps(kA, kB)
                afterModel.update_global_params(afterSS)
                assert afterModel.K == self.beforeModel.K - 1
                afterELBO = afterModel.calc_evidence(None, afterSS, None)
                gapAB = afterELBO - beforeELBO
                gap = self.beforeModel.calcHardMergeGap(self.beforeSS, kA, kB)
                print(gapAB)
                print(gap)
                assert np.allclose(gapAB, gap)
                assert np.allclose(gapAB, GapMat[kA, kB])
