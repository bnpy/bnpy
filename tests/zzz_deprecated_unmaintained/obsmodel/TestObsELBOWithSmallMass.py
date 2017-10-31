import numpy as np
import unittest
import copy

import bnpy


class Test(unittest.TestCase):

    """ Basic unit test for verifying ELBO gap before/after delete


    Each possible model depends on provided params K, N, D.
    Dataset is drawn from Normal(0, I) given provided seed.

    TODO: understand why current default values fail to yield positive gap!
    """

    def shortDescription(self):
        return None

    def __init__(self, testname,
                 obsModelName='ZeroMeanGauss',
                 K=2, N=100, D=1, seed=None, sF=1.0, xval=None,
                 Reps=0.01,
                 **kwargs):
        super(type(self), self).__init__(testname)
        self.K = K
        self.D = D
        self.N = N
        self.sF = sF
        self.Reps = Reps
        self.obsModelName = obsModelName
        if seed is None:
            self.seed = int(K * D)
        else:
            self.seed = int(seed)
        self.xval = xval

    def setUp(self):
        K = self.K
        N = self.N
        D = self.D

        # Randomly generate Data from standard normal (mean=0, var=1)
        PRNG = np.random.RandomState(self.seed)
        Data = bnpy.data.XData(PRNG.randn(N, D))

        if D == 1 and self.xval is not None:
            Data.X[0] = self.xval
        else:
            Data.X[0] = 0
        self.Data = Data

        PriorSpec = dict(ECovMat='eye', sF=self.sF, nu=3, kappa=1e-9)
        beforeModel = bnpy.obsmodel.\
            ObsModelConstructorsByName[self.obsModelName](
                'VB', D=D, **PriorSpec)
        afterModel = copy.deepcopy(beforeModel)
        after2Model = copy.deepcopy(beforeModel)

        # BEFORE
        # resp : NxK responsibilities
        #    last column is all zeros except for one entry
        resp = 1.0 / (K - 1) + 100 * PRNG.rand(N, K)
        resp = np.maximum(resp, 1e-40)
        resp[:, -1] = 1e-40
        resp[0, 0] = 1 - self.Reps
        resp[0, -1] = self.Reps
        resp /= resp.sum(axis=1)[:, np.newaxis]
        assert np.allclose(resp.sum(axis=1), 1.0)
        beforeLP = dict(resp=resp)
        beforeSS = beforeModel.get_global_suff_stats(Data, None, beforeLP)
        beforeModel.update_global_params(beforeSS)

        # AFTER
        # resp : Nx(K-1)
        #    original last column has been reassigned via the argmax rule
        kmax = resp[0, :-1].argmax()
        respNew = np.delete(resp, K - 1, axis=1)
        respNew[0, kmax] += resp[0, -1]
        assert np.allclose(respNew.sum(axis=1), 1.0)
        afterLP = dict(resp=respNew)
        afterSS = afterModel.get_global_suff_stats(Data, None, afterLP)
        afterModel.update_global_params(afterSS)

        # AFTER2
        # resp : NxK
        #    original last column still represented, but with zero mass
        resp2 = resp.copy()
        resp2[:, -1] = 0
        resp2[0, kmax] += resp[0, -1]
        assert np.allclose(resp2.sum(axis=1), 1.0)
        after2LP = dict(resp=resp2)
        after2SS = after2Model.get_global_suff_stats(Data, None, after2LP)
        after2Model.update_global_params(after2SS)

        self.beforeSS = beforeSS
        self.beforeModel = beforeModel
        self.afterSS = afterSS
        self.afterModel = afterModel
        self.after2SS = after2SS
        self.after2Model = after2Model

        self.beforeELBOalloc = calcAllocELBO(Data, resp)
        self.afterELBOalloc = calcAllocELBO(Data, respNew)
        self.respNew = respNew
        print('')
        print('K=%d  D=%d  N=%d  %s' % (K, D, N, self.obsModelName))
        print('----------------- setup complete.')

    def test_suff_stats_represent_same_whole_dataset(self):
        assert np.allclose(self.beforeSS.N.sum(),
                           self.afterSS.N.sum())
        assert np.allclose(self.beforeSS.xxT.sum(),
                           self.afterSS.xxT.sum())

    def test_ELBO_gap(self):
        beforeELBO = self.beforeModel.calcELBO_Memoized(self.beforeSS)
        afterELBO = self.afterModel.calcELBO_Memoized(self.afterSS)
        print("beforeELBO   % 9.6f" % (beforeELBO))
        print(" afterELB0   % 9.6f" % (afterELBO))
        assert afterELBO >= beforeELBO

        after2ELBO = self.after2Model.calcELBO_Memoized(self.after2SS)
        # print " after2ELB0  % 9.6f" % (after2ELBO)
        assert np.allclose(afterELBO, after2ELBO)

    def calc_ELBO_gap(self):
        beforeELBO = self.beforeModel.calcELBO_Memoized(self.beforeSS)
        afterELBO = self.afterModel.calcELBO_Memoized(self.afterSS)
        return afterELBO - beforeELBO

    def calc_ELBO(self):
        beforeELBO = self.beforeModel.calcELBO_Memoized(self.beforeSS)
        return beforeELBO


class TestRange(unittest.TestCase):

    """ Verify expected gap for a range of possible models.

    Each possible model depends on provided params K, N, D.
    Dataset is drawn from Normal(0, I) given provided seed.
    """

    def runTest(self):
        for obsModelName in ['Gauss', 'DiagGauss']:
            for K in [2]:
                for N in [1, 2, 4, 16, 100, 500, 5000]:
                    for D in [1, 2, 16, 32, 64]:
                        for seed in [2, 111, 222, 333]:
                            kwargs = dict(obsModelName=obsModelName,
                                          seed=seed,
                                          K=K, N=N, D=D)
                            curTest = Test("test_ELBO_gap", **kwargs)
                            curTest.setUp()
                            curTest.run()
                            curTest.tearDown()


def calcAllocELBO(Data, resp, gamma0=5):
    N = resp.sum(axis=0)
    SS = bnpy.suffstats.SuffStatBag(K=resp.shape[1])
    SS.setField('N', N, dims='K')

    from bnpy.allocmodel import DPMixtureModel
    model = DPMixtureModel('VB', dict(gamma0=gamma0))
    model.update_global_params(SS)
    L = model.calc_evidence(Data, SS, dict(resp=resp))
    return L


'''
class TestPlotGapVsXVals(unittest.TestCase):

    def runTest(self):
        suite = unittest.TestSuite()
        obsModelName = 'ZeroMeanGauss'
        xvals = np.linspace(-4, 4, 100)
        gaps = np.zeros_like(xvals)

        for ii, xval in enumerate(xvals):
            kwargs = dict(obsModelName=obsModelName,
                          xval=xval,
                          Reps=0.10,
                          K=2, N=1, D=1, sF=0.5)
            myTest = Test("calc_ELBO_gap", **kwargs)
            myTest.setUp()
            gaps[ii] = myTest.calc_ELBO_gap() / (myTest.N * myTest.D)
        from matplotlib import pylab
        pylab.plot(xvals, gaps, 'k.-')

        agap = myTest.afterELBOalloc - myTest.beforeELBOalloc
        pylab.plot(xvals, gaps + agap, 'r.-')
        pylab.ylim([-1, 10])
        pylab.title(obsModelName)
        pylab.xlabel('x')
        pylab.ylabel('ELBO gap')
        pylab.legend(['L_data', 'L_data + L_alloc + L_entropy'],
                     loc='center right')
        pylab.show(block=1)
'''
