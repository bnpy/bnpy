'''
'''
import numpy as np
import unittest
import copy

import bnpy
import ToyARK13


class TestNaturalParams(unittest.TestCase):

    def setUp(self, T=1000, K=1):
        Params = dict(
            ECovMat='eye',
            VMat='eye',
            MMat='eye',
            sF=1.23,
            sM=4.5,
            sV=0.321,
            nu=0)

        self.Data = ToyARK13.get_data(seed=12, nSeq=1, T=T)
        self.obsModel = bnpy.obsmodel.AutoRegGaussObsModel(
            'VB',
            Data=self.Data,
            **Params)
        SS = self.obsModel.calcSummaryStatsForContigBlock(self.Data, a=0, b=T)

        self.obsModel.update_global_params(SS)
        self.SS = SS

    def test_natural_to_common(self):
        nu, V, n_VMT, n_B = self.obsModel.calcNaturalPostParams(self.SS)

        oModel = copy.deepcopy(self.obsModel)
        oModel.convertPostToNatural()
        assert np.allclose(oModel.Post.nu, nu)
        assert np.allclose(oModel.Post.V, V)
        assert np.allclose(oModel.Post.n_VMT, n_VMT)
        assert np.allclose(oModel.Post.n_B, n_B)

        oModel.convertPostToCommon()
        assert np.allclose(oModel.Post.M, self.obsModel.Post.M)
        assert np.allclose(oModel.Post.V, self.obsModel.Post.V)
        assert np.allclose(oModel.Post.nu, self.obsModel.Post.nu)
        assert np.allclose(oModel.Post.B, self.obsModel.Post.B)
