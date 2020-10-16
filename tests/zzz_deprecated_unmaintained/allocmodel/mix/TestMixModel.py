'''
Unit-tests for FiniteMixtureModel.py
'''
import numpy as np
import bnpy
from bnpy.allocmodel import FiniteMixtureModel
from bnpy.suffstats import SuffStatBag


class TestMixModelEMUnifGamma(object):

    def shortDescription(self):
        return None

    def setUp(self):
        ''' Create simple case to double-check calculations.
        '''
        self.gamma = 1.0
        self.allocM = FiniteMixtureModel('EM', dict(gamma=self.gamma))
        self.N = np.asarray([1., 2., 3, 4, 5.])
        self.SS = SuffStatBag(K=5, D=1)
        self.SS.setField('N', self.N, dims='K')
        self.resp = np.random.rand(100, 3)
        self.precompEntropy = -1 * np.sum(
            self.resp * np.log(self.resp), axis=0)

    def test_update_global_params_EM(self):
        K = self.N.size
        self.allocM.update_global_params_EM(self.SS)
        wTrue = (self.N + self.gamma / float(K)  - 1.0)
        wTrue = wTrue / np.sum(wTrue)
        wEst = self.allocM.w
        print(wTrue)
        print(wEst)
        assert np.allclose(wTrue, wEst)

    def test_get_global_suff_stats(self):
        Data = bnpy.data.XData(np.random.randn(10, 1))
        SS = self.allocM.get_global_suff_stats(
            Data, dict(resp=self.resp),
            doPrecompEntropy=True)
        print(self.precompEntropy)
        print(SS.getELBOTerm('Hresp'))
        assert np.allclose(self.precompEntropy, SS.getELBOTerm('Hresp'))
        assert np.allclose(np.sum(self.resp, axis=0), SS.N)


class TestMixModelEMNonunifGamma(TestMixModelEMUnifGamma):

    def setUp(self):
        self.gamma = 2.0
        self.allocM = FiniteMixtureModel('EM', dict(gamma=self.gamma))
        self.N = np.asarray([1., 2., 3, 4, 5.])
        self.SS = SuffStatBag(K=5, D=1)
        self.SS.setField('N', self.N, dims='K')
        self.resp = np.random.rand(100, 3)
        self.precompEntropy = -1 * np.sum(
            self.resp * np.log(self.resp), axis=0)
