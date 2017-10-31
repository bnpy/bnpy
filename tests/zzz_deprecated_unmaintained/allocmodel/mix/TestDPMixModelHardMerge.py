'''
'''

import numpy as np
import unittest
import copy

import bnpy


class TestDPHardMerge(unittest.TestCase):

    def setUp(self):
        K = 5
        aModel = bnpy.allocmodel.DPMixtureModel(
            'VB',
            dict(
                gamma0=5,
                truncType='z'))
        SS = bnpy.suffstats.SuffStatBag(K=5, D=1)
        SS.setField('N', np.arange(K), dims='K')
        SS.setELBOTerm('Hresp', np.zeros(K), dims='K')
        SS.setMergeTerm('Hresp', np.zeros((K, K)), dims=('K', 'K'))
        aModel.update_global_params(SS)
        self.beforeModel = aModel
        self.beforeSS = SS
        self.beforeK = K

    def test_calcHardMergeGap(self):
        print('')
        beforeELBO = self.beforeModel.calc_evidence(None, self.beforeSS, None)

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
                assert np.allclose(gapAB, gap)
                assert np.allclose(gapAB, GapMat[kA, kB])
