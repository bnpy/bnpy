import copy
import numpy as np
import unittest
import bnpy
rho2beta_active = bnpy.util.StickBreakUtil.rho2beta_active

M_K4 = np.asarray([
    [100, 5, 5, 5],
    [5, 100, 5, 5],
    [5, 5, 100, 5],
    [5, 5, 5, 100],
])
Nfirst_K4 = np.asarray([1., 1., 1., 1.])


def pprintVec(xvec, fmt='%.3f'):
    print(' '.join([fmt % (x) for x in xvec]))


class TestHMMMergeConstructGlobals(unittest.TestCase):

    def shortDescription(self):
        return None

    def setUp(self):
        self.origK4_SS = bnpy.suffstats.SuffStatBag(K=4)
        self.origK4_SS.setField('TransStateCount', M_K4, dims=('K', 'K'))
        self.origK4_SS.setField('StartStateCount', Nfirst_K4, dims=('K'))

        self.origK4_aModel = bnpy.allocmodel.hmm.HDPHMM.HDPHMM(
            'VB',
            dict(gamma=10.0, alpha=1.0))
        self.origK4_aModel.update_global_params(self.origK4_SS)

        # Now perform a merge of the last two states (at indices 2 and 3)
        kA = 2
        kB = 3
        self.propK3_SS = self.origK4_SS.copy()
        self.propK3_SS.mergeComps(kA, kB)
        self.propK3_aModel = copy.deepcopy(self.origK4_aModel)
        self.propK3_aModel.update_global_params(self.propK3_SS, mergeCompA=kA,
                                                mergeCompB=kB)

    def test_verify_rho_valid(self):
        ''' Verify that the merge value of rho is sensible
        '''
        assert self.propK3_aModel.rho.size == 3
        assert np.all(self.propK3_aModel.rho >= 0)
        assert np.all(self.propK3_aModel.rho <= 1)

    def test_aaa_show_original_rho_beta(self):
        print('')
        print('rho')
        pprintVec(self.origK4_aModel.rho)
        print('')
        print('beta')
        pprintVec(rho2beta_active(self.origK4_aModel.rho))
        print('')
        print('omega')
        pprintVec(self.origK4_aModel.omega)

    def test_aaa_show_proposed_rho_beta(self):
        print('')
        print('rho')
        pprintVec(self.propK3_aModel.rho)
        print('')
        print('beta')
        pprintVec(rho2beta_active(self.propK3_aModel.rho))
        print('')
        print('omega')
        pprintVec(self.propK3_aModel.omega)

    def test_show__convergence_rho(self):
        ''' Observe what happens if we run several consecutive global updates

            Seems that in general, rho/omega/theta require many iters to be
            absolutely sure of convergence.
        '''
        print('')

        for iter in range(10):
            print('                         rho after %d global updates' % (iter))
            pprintVec(self.propK3_aModel.rho)

            self.propK3_aModel.update_global_params(self.propK3_SS)
            print((
                ' ' * 10) + '%.3f' % (self.propK3_aModel.OptimizerInfo['fval']))

    def test_show__convergence_rho_update_rho_only(self):
        ''' Observe what happens if we run several consecutive optimizer updates

            Seems that the optimizer does consistently find a fixed point
        '''
        print('')

        fval_orig = self.propK3_aModel.OptimizerInfo['fval']
        print((' ' * 10) + '%.3f' % (fval_orig))

        fvals = np.zeros(10)
        for iter in range(10):
            print('                         rho after %d optimizer updates' % (iter))
            pprintVec(self.propK3_aModel.rho)

            self.propK3_aModel.find_optimum_rhoOmega()
            fval = self.propK3_aModel.OptimizerInfo['fval']
            print((' ' * 10) + '%.3f' % (fval))
            fvals[iter] = fval

        # Verify that subsequent updates only improve on original fval
        assert fval_orig >= fvals[0]

        # Verify that we've converged
        assert np.allclose(fvals[0], fvals[1:])
